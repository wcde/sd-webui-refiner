# Refiner (webui Extension)
## Webui Extension for integration refiner in generation process

Extension loads from refiner checkpoint only UNET and replaces base UNET with it at last steps of generation.

## Usage

1. Activate extension and choose refiner checkpoint in extension settings on txt2img tab.
2. Set percent of refiner steps from total sampling steps.

Use Tiled VAE if you have 12GB or less VRAM.
