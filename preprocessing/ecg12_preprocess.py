"""
ECG 12-lead preprocessing pipeline.

This module provides ECG signal processing functionality including:
- QRS detection and extraction using RPNet detector
- Peak improvement and heart beat alignment  
- Signal filtering and aggregation
- Correlation-based flip detection for consistent lead orientation

Usage: python ecg12_preprocess.py or import ECG_Preprocessor class

Dependencies:
    - ecg_RPNet (external package)
    - scipy, numpy, pandas, matplotlib
    
Author: J. van der Valk
"""

import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy.signal as ssig
from ecg_RPNet import detector_RP
import matplotlib.pyplot as plt
import tqdm
import copy
import multiprocessing
import torch


class ECG_Preprocessor:
    """Class for processing 12-lead ECG signals with QRS detection and quality control."""
    
    def __init__(self, fs):
        self.fs = fs
        self.detector = detector_RP.Detector_RPNet()
        self.filt_count = 0
        self.mag_count = 0
        self.RP_fail_count = 0

    def eval_peaks(self, p, x):
        """Score R-peak placements by the mean inter-beat correlation.

        Extracts a 400-sample window (200 around each peak) for every
        beat and returns the correlation matrix plus its mean.
        """
        """Evaluate peak detection quality using correlation matrix."""
        xs = np.zeros([len(p[0]) - 2, 400])
        delete_first = False
        delete_last = False
        for jj, peak in enumerate(p[0][1:-1]):
            qrs = x[peak - 200 : peak + 200]
            if len(qrs) != 400:
                if jj == 0:
                    delete_first = True
                else:
                    delete_last = True
            else:
                xs[jj] = qrs

        if delete_first:
            xs = xs[1:]
        if delete_last:
            xs = xs[:-1]
        cc = np.corrcoef(xs)

        if type(cc) != np.ndarray:
            return cc, cc
        elif len(cc) < 2:
            print("potential error")
            return cc, 0
        return cc, np.mean(cc)

    def improve_peaks(self, ps, x):
        """Refine R-peak positions with a local search.

        For the least-correlated beat, slide the peak over neighbouring
        candidate maxima (within +/-100 samples) and keep the position
        that maximises the mean correlation.
    """
        """Improve peak positions using local search and correlation."""
        cc_og, mean_og = self.eval_peaks(ps, x)
        temp_ps = copy.deepcopy(ps)
        best_mean = mean_og
        if cc_og.ndim < 2:
            ccm = mean_og
        else:
            ccm = np.mean(cc_og, axis=1)
            
        for wp in np.argsort(ccm):
            wp += 1
            x_wp = ps[0][wp]

            ap = ssig.find_peaks(x)[0]
            ap = ap[ap < (x_wp + 100)]
            ap = ap[ap > (x_wp - 100)]
            best_pp = x_wp

            for pp in ap:
                temp_ps[0][wp] = pp
                cc, mean = self.eval_peaks(temp_ps, x)
                if mean > best_mean:
                    best_mean = mean
                    best_pp = pp

            temp_ps[0][wp] = best_pp

        return temp_ps

    def split_hb2(self, x_ecg, first=True):
        """Detect QRS complexes and split the ECG into heart beats.

        Runs RPNet, refines peaks, and returns (beats [N,400], rri
        intervals, exclusion info). Returns (0,0,0) if too few peaks.
        """
        """Split ECG into individual heart beats."""
        ecg, peaks, y = self.detector.predict(x_ecg.reshape(1, -1), calibrate=True)

        if len(peaks[0]) < 3:
            print("Too little peaks")
            return 0, 0, 0
        
        temp_ps = copy.deepcopy(peaks)
        for _ in range(5):
            temp_ps = self.improve_peaks(temp_ps, x_ecg)

        xs = np.zeros([len(temp_ps[0]) - 2, 400])
        skip = 0
        
        for i, peak in enumerate(temp_ps[0][1:-1]):
            qrs = x_ecg[peak - 200 : peak + 200]
            if len(qrs) != 400:
                if i == 0:
                    skip = 1
                else:
                    rri = temp_ps[0][1:-1] - temp_ps[0][:-2]
                    begin = temp_ps[0][0]
                    end = 5000 - sum(rri) - begin
                    return xs[:-1], rri, [begin, end]
            else:
                xs[i] = qrs

        rri = temp_ps[0][(1 + skip):] - temp_ps[0][skip:-1]
        begin = temp_ps[0][skip]
        end = 5000 - sum(rri) - begin
        return xs[skip:], rri, [begin, end]

    def return_qrs(self, qrs_in, rri_in, label, aggregate=True):
        """Aggregate heart-beat windows into median/mean/std feature rows."""

        """Return QRS aggregation methods."""
        if aggregate:
            return np.median(qrs_in, axis=0), np.mean(qrs_in, axis=0), np.std(qrs_in, axis=0), np.mean(rri_in), np.std(rri_in), label
        else:
            return qrs_in, np.std(qrs_in, axis=0), np.mean(rri_in), np.std(rri_in), label

    def _checkConsecutive(self, arr, max_val=5):
        """Detect runs of consecutive integers in an array (returns the run or False)."""

        """Check for consecutive elements in array."""
        if len(arr) < max_val:
            return False
        _count = 0
        _max_count = 0
        _end = 0
        for i, d in enumerate(np.diff(arr)):
            if d == 1:
                _count += 1
                if _count > _max_count:
                    _max_count = _count
                    _end = i + 2
            else:
                _count = 0
        if _max_count > 4:
            return arr[_end-5:_end]
        else:
            return False

    def check_overlap(self, begin1, begin2, rri1, rri2, start2):
        """Find where two R-peak sequences start to overlap (index i)."""

        """Check overlap between heart beat sequences."""
        l = begin1 + sum(rri1) - start2
        i = 0
        while begin2 < l:
            begin2 += rri2[i]
            i += 1
        return i

    def checkNeg_sigs(self, qrs):
        """Classify polarisation of QRS complexes.
        Returns sure/poss "correct" or "incorrect" (normal vs inverted R).
        """
        """Check for negative magnitude signals and categorize quality."""
        mag_count = 0
        sure_count = 0
        for i, hb in enumerate(qrs):
            hb_in = hb - np.median(hb)
            if 4 * abs(np.max(hb_in[150:250])) < abs(np.min(hb_in[150:250])):
                mag_count -= 1
                if abs(np.max(hb_in[100:150])) < abs(np.min(hb_in[100:150])):
                    sure_count -= 1
            else:
                mag_count += 1
                if abs(np.max(hb_in[100:150])) > abs(np.min(hb_in[100:150])):
                    sure_count += 1
        if sure_count > 1:
            return "sure_correct"
        elif sure_count < -1:
            return "sure_incorrect"
        
        if mag_count > 0:
            return "poss_correct"
        else:
            return "poss_incorrect"

    def embed_sig2(self, sig_unfilt, fs=None, aggregate=True):
        """Full pipeline for one lead window: filter -> split beats ->
        quality-filter -> aggregate into median/mean/std QRS + RRI.
        Returns (median, mean, std, mean_rri, std_rri, status).
        """
        """Embed and process ECG signal into heart beat features."""
        if fs is None:
            fs = self.fs
            
        if fs == 1000:
            sig_unfilt = ssig.resample(sig_unfilt, 5000)
            fs = 500
        elif fs == 300:
            sig_unfilt = ssig.resample(sig_unfilt, 15000)
            fs = 500

        filt = ssig.firwin(
            len(sig_unfilt),
            cutoff=[0.5, 45],
            window="blackmanharris",
            pass_zero="bandpass",
            fs=fs,
        )
        sig = ssig.convolve(sig_unfilt, filt, mode="same", method="auto")

        sig1 = sig[:5000]

        if len(sig) > 6000:
            sig2 = sig[-5000:]

            qrs1, rri1, ex1 = self.split_hb2(sig1)
            qrs2, rri2, ex2 = self.split_hb2(sig2)

            if type(qrs1) == int:
                if type(qrs2) == int:
                    self.RP_fail_count += 1
                    return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"
                else:
                    qrs = qrs2
            else:
                if type(qrs2) == int:
                    qrs = qrs1
                elif len(sig) >= 10000:
                    qrs = np.concatenate([qrs1, qrs2])
                    rri = np.concatenate([rri1, rri2])
                else:
                    count = ex2[1] + 10
                    i = 0
                    while count < (len(sig) - 5000 + ex1[1]):
                        i += 1
                        if i == len(rri2):
                            break
                        count += rri2[-i]
                    qrs = np.concatenate([qrs1[3:], qrs2[-i:]])
                    rri = np.concatenate([rri1[4:], rri2[-i:]])
        else:
            qrs, rri, ex = self.split_hb2(sig1)
            if type(qrs) == int:
                self.RP_fail_count += 1
                return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"

        qrs_arg = []
        for i, hb in enumerate(qrs):
            if np.mean(abs(hb)) > 1:
                qrs_arg.append(i)

        qrs = qrs[qrs_arg]
        rri = rri[qrs_arg]

        if len(qrs) == 0:
            self.mag_count += 1
            return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"
        elif len(qrs) == 1:
            self.mag_count += 1
            return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"

        cc = np.corrcoef(qrs)

        try:
            filt1 = cc.mean(axis=1) > 0.85
        except Exception as e:
            print(f"Error: {e}")
            return (
                np.median(qrs, axis=0),
                np.mean(qrs, axis=0),
                np.std(qrs, axis=0),
                np.mean(rri),
                np.std(rri),
                "error",
            )

        qrs_filt = qrs[filt1]

        if len(qrs_filt) == 0:
            self.filt_count += 1
            return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"
        elif len(qrs_filt) == 1:
            self.filt_count += 1
            return np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, "empty"

        return self.return_qrs(qrs_filt, rri, "normal", aggregate=aggregate)

    def check_negative(self, ecg):
        """Drop per-patient rows whose positive/negative QRS disagree (<0.5 corr)."""

        """Check for negative correlated signals."""
        for pi in ecg.PatientUniqueID.unique():
            df_pi = ecg.loc[ecg.PatientUniqueID == pi]
            df_pi.drop_duplicates(subset=["METING_DATUM"], keep="first", inplace=True)
            df_ecgs_p = df_pi.qrsw_p.values
            df_ecgs_n = df_pi.qrsw_n.values
            
            for i, ecg_p in enumerate(df_ecgs_p):
                ecg_n = df_ecgs_n[i]
                if type(ecg_p) == np.ndarray and type(ecg_n) == np.ndarray:
                    cc = np.corrcoef(ecg_p, ecg_n)
                    if cc[0,1] < 0.5:
                        df_pi.drop(df_pi.index[i], inplace=True)

    def get_corr_df(self, df):
        """Stack positive+negative QRS columns and build their correlation matrix."""

        """Get correlation matrix for ECG signals."""
        qrsws = np.stack(df.qrsw_p.values)
        qrsws_n = np.stack(df.qrsw_n.values)
        qrss = np.concatenate([qrsws, qrsws_n])
        
        corrs = np.nan_to_num(np.corrcoef(qrss), nan=-1)
        return qrsws, qrsws_n, corrs

    def flip(self, df, indices):
        """Swap qrsw_p/qrsw_n (flip lead orientation) for the given row indices."""
        idx = list(df.index[indices])
        # pandas-3: multi-row `df.loc[list, col] = <array>` hits a split-path
        # that fails when other columns hold numpy arrays (object dtype).
        # Assign cell-by-cell with .at, which is portable across pandas 2+3
        # and works on array-valued columns.
        for i in idx:
            df.at[i, "flipped"] = not bool(df.at[i, "flipped"])
        for i in idx:
            a = df.at[i, "qrsw_p"]
            b = df.at[i, "qrsw_n"]
            df.at[i, "qrsw_p"] = b
            df.at[i, "qrsw_n"] = a
        return df

    def check_and_accept(self, df, df_best, cor_best, start=False):
        """Accept a flip candidate only if its mean correlation improves
        over the current best (else keep the previous state).
        """
        """Check correlation and accept/reject processing."""
        qrsws, _, corrs = self.get_corr_df(df)
        cor_mean = (np.mean(corrs[:qrsws.shape[0], :qrsws.shape[0]]) + 0.9*np.mean(corrs[qrsws.shape[0]:, qrsws.shape[0]:])) / 2
        print("Correlation mean:", cor_mean)

        if start:
            return cor_mean
        else:
            if cor_mean > (cor_best + 0.00001):
                print(f"Accepted....{cor_mean}")
                return df, cor_mean
            else:
                print(f"Rejected....{cor_mean} vs {cor_best}")
                return df_best, cor_best

    def flip_based_on_correlation(self, df, n=5, last=False, first=False, median=False):
        """Iteratively re-orient individual leads to maximise cross-lead
        correlation (multiple strategies: last, first, median, default).
        """
        """Flip signals based on correlation analysis."""
        df_flipped = df.copy()
        for _ in range(n):
            qrsws, qrsws_n, corrs = self.get_corr_df(df_flipped)

            if last:
                corrs_p1 = corrs[-1, qrsws.shape[0]:]
                corrs_p2 = corrs[-1, :qrsws.shape[0]]
            elif first:
                corrs_p1 = corrs[0, qrsws.shape[0]:]
                corrs_p2 = corrs[0, :qrsws.shape[0]]
            elif median:
                corrs_p2 = np.mean(corrs[:qrsws.shape[0], :qrsws.shape[0]], axis=1)
                corrs_p1 = np.zeros(corrs_p2.shape)
            else:
                corrs_p1 = np.mean(corrs[:qrsws.shape[0], :qrsws.shape[0]], axis=1)
                corrs_p2 = np.mean(corrs[:qrsws.shape[0], qrsws.shape[0]:], axis=1)

            pick_n = corrs_p2 >= corrs_p1
            df_flipped = self.flip(df_flipped, pick_n)
            print(f"{sum(pick_n)} flips made")
            if sum(pick_n) == 0:
                break

        return df_flipped

    def flip_zero_signals(self, df):
        """Flip rows whose positive QRS trace is all zeros."""

        """Flip zero signals."""
        df_flipped = df.copy()
        pick_n = [i[0] == 0 for i in df_flipped.qrsw_p.values]
        df_flipped = self.flip(df_flipped, pick_n)
        print(f"{sum(pick_n)} zero flips made")
        return df_flipped

    def delete_entries(self, df):
        """Remove duplicate recordings made within 5 minutes of each other."""

        """Delete duplicate entries within 5 minutes."""
        df = df.sort_values("METING_DATUM")
        df["METING_DATUM"] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
        df["delete"] = False
        for i in range(0, len(df) - 1):
            if df.loc[i + 1, "METING_DATUM"] - df.loc[i, "METING_DATUM"] < pd.Timedelta(minutes=5):
                df.loc[i, "delete"] = True
        df = df.query("delete == False")
        return df

    def filter_dataset(self, df_ecg):
        """Per-patient orientation cleanup: deduplicate, zero-flip,
        correlation-based flips, and propagate the chosen orientation
        back to the raw dataframe.
        """
        """Filter and preprocess entire dataset."""
        for pi in df_ecg.PatientUniqueID.unique():
            df_pre = df_ecg.query("PatientUniqueID == @pi")
            df = df_pre[["METING_DATUM", "qrsw_p", "qrsw_n"]].groupby(by="METING_DATUM").mean()

            df = self.delete_entries(df)
            df["flipped"] = False

            qrsws, qrsws_n, corrs = self.get_corr_df(df)
            cor_best = (np.mean(corrs[:qrsws.shape[0], :qrsws.shape[0]]) + 0.9*np.mean(corrs[qrsws.shape[0]:, qrsws.shape[0]:])) / 2

            df_temp = self.flip_zero_signals(df)
            _, cor_best = self.check_and_accept(df_temp, df, cor_best)

            df_temp = self.flip_based_on_correlation(df, n=5)
            _, cor_best = self.check_and_accept(df_temp, df, cor_best)

            df_temp = self.flip_based_on_correlation(df, n=1, last=True)
            df_temp = self.flip_based_on_correlation(df_temp, n=5)
            _, cor_best = self.check_and_accept(df_temp, df, cor_best)

            df_temp = self.flip_based_on_correlation(df, n=1, median=True)
            df_temp = self.flip_based_on_correlation(df_temp, n=5)
            _, cor_best = self.check_and_accept(df_temp, df, cor_best)

            df = df[[False if x[0] == 0.0 else True for x in df.qrsw_p]]

            for date in df_pre.METING_DATUM.unique():
                for ind in df_ecg.query("PatientUniqueID == @pi").query("METING_DATUM == @date").index:
                    if date in df.index:
                        index = df.query("METING_DATUM == @date").index
                        if len(index) > 0 and df.loc[index[0], "flipped"]:
                            df_ecg.at[ind, "qrsw_n"] = df_ecg.loc[ind, "qrsw_p"]
                            df_ecg.at[ind, "qrsw_p"] = df.loc[date, "qrsw_n"]
                    else:
                        df_ecg.drop(ind, inplace=True)
        return df_ecg

    def calc_embed_sep(self, ecg, ecg_labeled, leads=12, ecg_str="ecg12", date_str="Date", pi_str="PatientID", aggregate=False):
        """Loop over rows and embed each of the 12 leads separately.

        Builds the per-row feature arrays (median/mean/std QRS + RRI per lead)
        and stores them on ecg_labeled as whole columns. Column assignment is
        deferred to after the loop because pandas 3 does not allow assigning a
        2D array via .at row-by-row into a new column.
        """
        max_ind = 0
        ecg_no_ag = pd.DataFrame(columns=ecg_labeled.columns)
        ecg_no_ag.drop("SIGNAL", axis=1, inplace=True)

        n_rows = len(ecg_labeled)
        # Pre-allocate per-row feature matrices.
        median_qrs = np.zeros((n_rows, leads, 400))
        mean_qrs = np.zeros((n_rows, leads, 400))
        std_qrs = np.zeros((n_rows, leads, 400))
        rri_mean = np.zeros((n_rows, leads))
        rri_std = np.zeros((n_rows, leads))

        for row_i, j in enumerate(tqdm.tqdm(ecg_labeled.index)):
            sigs = ecg.loc[j, ecg_str]
            pi = ecg.loc[j, pi_str]
            date = ecg.loc[j, date_str]

            for i in range(leads):
                sig = sigs[:, i]
                if i == 3:
                    sig = -sig
                qrs_median, qrs_mean, qrs_std, rri_mean_i, rri_std_i, label = self.embed_sig2(sig)
                median_qrs[row_i, i] = qrs_median
                mean_qrs[row_i, i] = qrs_mean
                std_qrs[row_i, i] = qrs_std
                rri_mean[row_i, i] = rri_mean_i
                rri_std[row_i, i] = rri_std_i

        # Assign as whole columns (pandas-3 safe).
        ecg_labeled["median qrs"] = list(median_qrs)
        ecg_labeled["mean qrs"] = list(mean_qrs)
        ecg_labeled["std qrs"] = list(std_qrs)
        ecg_labeled["mean rri"] = list(rri_mean)
        ecg_labeled["std rri"] = list(rri_std)

        return ecg_labeled


def run():
    """CLI entry point (placeholder run of the pipeline)."""

    """Run the preprocessing pipeline."""
    print("ECG Preprocessor - Run pipeline")


if __name__ == "__main__":
    run()
