# Photorealistic Style Transfer 🖼️📸

A high-quality neural style transfer implementation optimized for M2 Macs, producing photorealistic results while maintaining the content's structure and the style's characteristics.

## Features

- Optimized for M2 Macs using Metal Performance Shaders (MPS)
- High-quality photorealistic style transfer
- Real-time progress visualization
- Interactive loss tracking
- Support for high-resolution images (up to 1024px)
- Quality-optimized parameters for best results

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/photorealistic_style_transfer.git
cd photorealistic_style_transfer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your content and style images through the web interface.

3. Adjust parameters if needed:
   - Content Weight: Controls content preservation (0.1-5.0)
   - Style Weight: Controls style transfer strength (1e4-2e5)
   - TV Weight: Controls image smoothness (1e-4-1e-2)

4. Click "Start Style Transfer" and wait for the process to complete.

## Default Parameters

The M2-optimized version uses the following default parameters:
- Steps: 500
- Content Weight: 0.3
- Style Weight: 3e5
- TV Weight: 5e-8
- Max Size: 1024px
- Learning Rate: 0.02

## Performance

- Processing time: ~5-10 minutes for 500 steps on M2 Mac
- Memory usage: Optimized for M2's unified memory architecture
- Quality: High-resolution output with photorealistic results

## Requirements

- macOS with M1/M2 chip
- Python 3.8+
- PyTorch with MPS support
- See requirements.txt for full dependency list

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original Neural Style Transfer paper by Gatys et al.
- Optimized for M2 Macs using PyTorch's MPS backend
- Uses VGG19 for feature extraction 
