# GGUF Audit & Metadata Extractor

A lightweight, fully autonomous, and dependency-free pure Python tool for forensic analysis, metadata extraction, and security auditing of GGUF models.

## Key Features
- **Offline Binary Structure Parsing:** Reads metadata and tensors without downloading full weights (utilizes `mmap` memory mapping or HTTP Range requests).
- **Template Extraction:** Automatically extracts and saves conversational Jinja templates (`.jinja`).
- **Autonomous Security Audit:** Static metadata inspection, tokenizer vocabulary validation, anomaly detection, and architecture mapping verification.
- **Modern HTML Report:** Automatically generates a clean, responsive HTML report with a dark-mode design—completely self-contained without external JavaScript dependencies.


## Security Overview & Forensic Threat Model

While modern model formats (`.gguf` and `.safetensors`) were designed to mitigate legacy arbitrary code execution vulnerabilities (such as PyTorch `pickle` deserialization RCE), they introduce distinct architectural, structural, and behavioral risk vectors across local and agentic execution environments.


### GGUF & Agentic Execution Attack Vectors

GGUF (GPT-Generated Unified Format) stores key-value metadata alongside quantized tensors. Although loading raw GGUF binaries does not trigger Python `pickle` evaluation, the ecosystem presents critical operational threat vectors:

- **C/C++ Parser Memory Corruption:** Inference engines (such as `llama.cpp` and derived runtimes like Ollama or LM Studio) parse GGUF binary headers in low-level memory space. Crafted headers containing corrupted array bounds, malformed strings, or mismatched tensor dimensions can trigger buffer overflows or Out-of-Bounds memory operations, opening exploitation vectors on the host system.
- **Metadata & Quantization Fraud:** Adversaries modify header strings (`general.name`, `general.size_label`, parameter counts) in lightweight models (e.g., 0.5B parameters) to mimic flagship high-parameter releases (e.g., 70B+ architectures). This enables download farming, cache exhaustion, and vectoring users toward malicious auxiliary repos.
- **Embedded Chat Template Hijacking:** GGUF binaries contain raw Jinja conversation templates (`tokenizer.chat_template`). Injected templates silently alter system prompts, neutralize model alignment, or append unseen system directives to every user query.
- **Agentic Weaponization (Tool Calling Exploitation):** When a GGUF model is deployed within an agentic workflow granting tool access (Python REPL, shell execution, local file I/O, web scraping), compromised weights or poisoned metadata force the model to issue unauthorized tool calls. This transforms the model into an active vector for secondary malware retrieval, local data exfiltration, and host-level Remote Code Execution (RCE).


### Addendum


### Attack Vectors & Vulnerabilities in `.safetensors`

GGUF Auditor does not test safetensors.

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

