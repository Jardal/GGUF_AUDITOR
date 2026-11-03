# GGUF Audit & Metadata Extractor

A lightweight, fully autonomous, and dependency-free pure Python tool for forensic analysis, metadata extraction, and security auditing of GGUF models.

## Key Features
- **Offline Binary Structure Parsing:** Reads metadata and tensors without downloading full weights (utilizes `mmap` memory mapping or HTTP Range requests).
- **Template Extraction:** Automatically extracts and saves conversational Jinja templates (`.jinja`).
- **Autonomous Security Audit:** Static metadata inspection, tokenizer vocabulary validation, anomaly detection, and architecture mapping verification.
- **Modern HTML Report:** Automatically generates a clean, responsive HTML report with a dark-mode design—completely self-contained without external JavaScript dependencies.


### 2. Attack Vectors & Vulnerabilities in `.safetensors`

While the `.safetensors` format eliminates executable bytecode (`pickle` opcodes) present in legacy PyTorch formats (`.bin`, `.pt`), structural and pipeline vulnerabilities still exist:

1. **Header Overflow / DoS:** Vector 1: Parser Divergence & Memory Exhaustion (DoS).
* **Mechanism:** The format relies on a JSON header defining tensor shapes/sizes followed by raw data. Non-standard or third-party parsers with weak validation can be exploited via specially crafted files declaring massive header lengths, triggering instant Out-Of-Memory (OOM) conditions.
* **Impact:** Denial of Service (DoS) on inference servers or local desktop runners.


2. **Compromised Upstream Conversion:** Vector 2: Conversion Pipeline Exploitation.
* **Mechanism:** While the final container is inert, the *conversion pipeline* (automating translation from legacy `.bin` formats to `.safetensors`) often executes arbitrary code or untrusted scripts on remote repository builders or local environments.
* **Impact:** Remote Code Execution (RCE) on build servers, credential exfiltration, and infrastructure compromise.


3. **Configuration Manipulation:** Vector 3: Metadata & Tokenizer Poisoning.
* **Mechanism:** Tensor weights remain pure data, but accompanying JSON headers, configurations (`config.json`), and tokenizers can be manipulated to remap special tokens or inject adversarial chat templates.
* **Impact:** Safety filter bypass (jailbreaks), context corruption, and targeted prompt injections.


4. **Logical Backdoors:** Vector 4: Training-Time Weight Poisoning.
* **Mechanism:** The format guarantees numeric validity, not semantic intent. Models can be adversarially fine-tuned to pass standard benchmarks while embedding triggers that activate specific malicious behaviors or payload generations when deployed.
* **Impact:** Deterministic, stealthy execution of backdoored logic completely invisible to static structural scanners.
