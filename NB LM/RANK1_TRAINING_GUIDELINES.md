# Rank 1 Training Guidelines & Architecture

To ensure future model trainings—including **Deep Neural Networks (DNNs)**, **K-Nearest Neighbors (KNNs)**, and **Graph Convolutional Networks (GCNs)**—achieve the **Rank 1 training benchmark** and maximize the competition's specific evaluation metric, developers and engineers must adhere to the following architectural and biological guidelines.

### **1. Performance: The "Fast Path" GPU Architecture**
The primary technical failure identified in recent diagnostics was the "Slow Path" bottleneck where the CPU and GPU synchronized on every batch.
*   **Eliminate "Kinks in the Hose":** Never call `.get()`, `.asnumpy()`, or `np.asarray()` inside a batch loop. This creates a high-latency "handshaking" overhead that can turn an 8-minute inference into 80 minutes.
*   **Use Asynchronous Pipelining:** Pre-allocate GPU-resident buffers (using `cp.zeros`) to act as a **"Conveyor Belt."** The A100 should queue matrix multiplications (GEMMs) and only perform a single, massive transfer to host RAM at the end of a full data chunk.
*   **Manual GEMM for Inference:** For RAPIDS cuML models (like LogReg or KNN), use manual matrix multiplication (`cp.dot(x, W.T) + b`) in inference helpers to bypass Python overhead and potential `AttributeErrors`.

### **2. Metric Focus: IA-Weighted Accuracy**
The competition is graded on the **Information Accretion (IA) weighted F1 measure**, which rewards the prediction of rare, specific GO terms.
*   **Mandatory IA Class Weights:** All future trainings (especially DNNs and LogReg) must utilize IA weights as `class_weight` during the `.fit()` call. This forces the model to prioritize high-value, deep-hierarchy terms.
*   **Aspect-Specific Thresholding:** Do not use a single global probability threshold (e.g., 0.40) for all predictions. Experimental data from the KNN baseline shows that tuning thresholds separately for MF, BP, and CC (e.g., using a lower threshold for harder-to-predict BP terms) provides a **+3.3% immediate boost** to the F1 score.

### **3. Strategic Modeling: Modular Stacking (Phase 2 & 3)**
The "Winning Edge" is not found in a single model but in the **Modular Stacking** of statistical and graph-based models.
*   **DNN Strategy (Brute Force Blending):** For the DNN, ingest all seven multimodal feature inputs (T5, ESM2-Large, Ankh, Taxa, and Text embeddings). Perform extreme ensembling by averaging predictions across **25 different models** (5-fold CV across 5 random states) to stabilize noisy predictions.
*   **GCN Strategy (Structural Stacking):** Do **not** feed raw protein sequences or embeddings into the GCN. Instead, feed the **Out-Of-Fold (OOF) predictions** from Level 1 (LogReg, GBDT, DNN). The GCN’s role is to structure these statistical features according to the biological logic of the GO graph.
*   **Ontology Specialization:** Always train three **independent models**—one each for Biological Process (BP), Molecular Function (MF), and Cellular Component (CC)—to respect their distinct hierarchical structures.

### **4. Biological Integrity: Hierarchy Enforcement (Phase 4)**
A state-of-the-art statistical model will fail if its predictions violate biological rules.
*   **Post-Hoc Enforcement:** Do not try to "bake in" hierarchical constraints during training using custom loss functions. Winners found that **strict post-processing** using **Max Propagation (Parent Rule)** and **Min Propagation (Child Rule)** is more robust.
*   **Propagation Rules:**
    *   **Max Prop:** If a child term is predicted, ensure all ancestors have a score at least as high.
    *   **Min Prop:** If a parent term has a low score, its specific children cannot have a higher score.

### **5. Engineering and Environment**
*   **Resource Planning:** Training base models and GCN stackers requires **32 GB of GPU RAM** (e.g., V100/A100).
*   **Artifact Synchronization:** Every training phase must conclude with a `STORE.maybe_push` to register OOF predictions and term definitions (`top_terms_{asp}.json`). This ensures Phase 3 (GCN) can correctly "pull" the features it needs to start training.

**Analogy for Future Success:**
Our previous mistakes were like trying to win a race by having the driver stop every 10 feet to check the engine (per-batch syncing). Going forward, we build a **high-speed conveyor system** (GPU Pipelining), fuel it with **high-octane IA-weighted data**, and then use a **structural architect** (GCN/Post-processing) to ensure the final building matches the biological blueprint (GO Hierarchy).
