@echo off
REM Decoder Ring standalone Ollama island - RTX 2070 SUPER, port 11436.
REM Env lines REPLICATED from foreman "Start 2070 Ollama.bat" (copied, NOT called:
REM decoder-ring has zero runtime dependency on the foreman tree).
REM  - Pinned by GPU UUID (nvidia-smi -L), NOT integer index: index mismapped live
REM    2026-07-09 and loaded onto the 4080. Update the UUID if the 2070 is replaced.
REM  - OLLAMA_LLM_LIBRARY=cuda_v13 is LOAD-BEARING: this Ollama build has a Vulkan
REM    backend that IGNORES CUDA_VISIBLE_DEVICES; without the pin the 4080 re-enters
REM    as Vulkan0 and models land on it anyway.
REM  - Model store is user-level and shared: llama3.2:1b (already pulled) is visible
REM    here with no new pull.
REM  - Coexists with the foreman island on 11435 (same 8GB card): gemma3:4b ~3.6GB +
REM    llama3.2:1b ~1.5GB fit. If VRAM is ever tight Ollama part-offloads to CPU
REM    (slower, but never the wrong GPU).
REM  - To STOP: close this window (kills the whole tree). Killing the serve PID alone
REM    orphans the child llama-server.exe runner, which keeps ~1.5GB resident on the 2070
REM    while /api/ps shows empty - if you must kill by PID use `taskkill /PID <pid> /T /F`.
set OLLAMA_HOST=127.0.0.1:11436
set OLLAMA_LLM_LIBRARY=cuda_v13
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=GPU-1fc6caa7-b318-ef36-b655-7eeed44cd9cc
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve
