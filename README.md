# Clustering Analysis

This repository contains a set of scripts for analyzing Ca²⁺ imaging, ventral root recordings, and stimulus data. Follow the steps below in order:

---

### 1. **Metadata Extraction**
**Script:** `get_meta_data_from_sweeps.py`  
**Description:** Collects metadata from Ca imaging recordings, ventral root data, and stimulus logs.  
**Outputs:**  
- `meta_data.csv`  
- `sampling_rate.csv`

---

### 2. **Match Sampling Rates**
**Script:** `match_sampling_rates.py`  
**Description:** Aligns the sampling rates of various recordings for consistency.

---

### 3. **Select Good Sweeps**
**Script:** `select_good_sweeps.py`  
**Description:** Filters high-quality recordings for further analysis and converts them to ΔF/F.

---

### 4. **Stimulus Preprocessing**
**Script:** `stimulus_pre_processing.py`  
**Description:** Processes stimulus data for integration in the analysis pipeline.

---

### 5. **Ventral Root: Preprocessing**
**Script:** `ventral_root_pre_processing.py`  
**Description:** Converts Olympus `.txt` files and prepares ventral root recordings.

---

### 6. **Ventral Root: Time Alignment**
**Script:** `ventral_root_align_recordings_with_ca_imaging.py`  
**Description:** Aligns ventral root recordings with Ca imaging data temporally.

---

### 7. **Ventral Root: Event Detection**
**Script:** `ventral_root_event_detection.py`  
**Description:** Detects swim bouts and other motor events in ventral root traces.

---

### 8. **Linear Scoring of Responses**
**Script:** `linear_scoring_analysis.py`  
**Description:**  
Applies linear regression to score ROI responses:  
**Score = R² × slope**

---

### Optional
- **Remove Motor Events from Ca Responses**  
  `remove_motor_events_from_ca_responses.py`

---

**Author:** Nils Brehm — 2024