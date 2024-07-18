# AI-Infra Benchmarks

Welcome to the **AI-Infra Benchmarks** repository. This repository serves as a centralized location for benchmarks related to Large Language Model (LLM) inference and other performance evaluations. As the repository evolves, we will incorporate additional benchmarks for fine-tuning and inference to gauge performance across different hardware configurations.

## Directory Structure

- **`/nvidia/`**: Contains benchmarks specific to Nvidia hardware.
- **`/common/`**: Includes benchmarks that are not specific to any hardware, such as serving benchmarks.
- **`/amd/`**: Will contain benchmarks for running on AMD hardware.

## How to Run Benchmarks

1. **Clone this repository:**

   ```bash
   git clone https://github.com/runpod/ai-infra-benchmarks.git
   cd ai-infra-benchmarks


2. **Follow instructions in the respective hardware directory to run the benchmarks.**

   - For Nvidia benchmarks, navigate to the `/nvidia/` directory and follow the provided instructions.
   - For common benchmarks, check the `/common/` directory.
   - For AMD benchmarks, follow the instructions in the `/amd/` directory.

## Notes

1. This repository currently contains inference benchmarks only. As our requirements evolve, we will actively update the repository to include additional benchmarks and tools.

## Contributing
Please follow the standard pull request process and include relevant details about the benchmarks or improvements you are adding.
