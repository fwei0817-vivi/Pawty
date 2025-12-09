Pet Style Generation Service - Integration Guide
Overview
This service transforms pet photos into specific artistic styles using DreamShaper 8 (a fine-tuned Stable Diffusion 1.5 model). It runs locally and generates high-aesthetic outputs.
Architecture
●	Model: Lykon/dreamshaper-8 (Auto-downloaded on first run ~4GB).
●	Scheduler: DPMSolver++ (Optimized for speed and sharpness).
●	Input: Local file path (JPEG/PNG).
●	Output: Local file path (saved image).
Setup Instructions
1.	System Requirements:
○	OS: Linux (Ubuntu 20.04+), Windows, or macOS.
○	GPU: NVIDIA GPU (8GB+ VRAM) recommended for <5s generation.
○	CPU/Mac: Supported but slower (30s - 3mins).
2.	Installation:
chmod +x setup_env.sh
./setup_env.sh

Python API Contract
1. Initialization
Initialize the service once at application startup.
from pet_style_service import PetStyleService

# Initialize once (loads model into VRAM)
service = PetStyleService()

2. Transformation
The transform_image method now requires a species parameter to prevent species-swapping (e.g., turning a dog into a cat).
output_path = service.transform_image(
    input_image_path="/tmp/uploads/user_photo.jpg",
    output_path="/tmp/results/processed_image.png",
    style_prompt="Cyberpunk style, neon lights",
    species="dog",    # CRITICAL: 'dog', 'cat', 'hamster', etc.
    strength=0.75     # Optional: 0.6 to 0.8 is the sweet spot
)

Parameter	Type	Default	Description
input_image_path	str	Required	Absolute path to the source image.
output_path	str	Required	Absolute path where the result will be saved.
style_prompt	str	Required	The artistic style (e.g., "Pixar style").
species	str	"dog"	New: The animal type. Helps the AI maintain identity.
strength	float	0.75	Creativity level (0.0 - 1.0). Higher = more style, less original structure.
3. Error Handling
The method returns None if generation fails.
result = service.transform_image(...)
if result is None:
    # Handle error (e.g., return 500 status to frontend)
    print("Image generation failed.")

Troubleshooting Common Issues
●	final_sigmas_type zero is not supported: This means diffusers is trying to use an incompatible scheduler algorithm. Ensure you are using the provided pet_style_service.py which explicitly sets algorithm_type="dpmsolver++".
●	"It looks like a human": Reduce strength to 0.65 or ensure species is set correctly. The service has built-in negative prompts to suppress human features.
●	Slow Performance: Ensure PyTorch is using CUDA.
import torch
print(torch.cuda.is_available()) # Should be True

