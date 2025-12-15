# Neural Chess Engine

A Python chess engine powered by a convolutional neural network (CNN) with a GUI using Pygame.  
The engine learns value estimation through self-play and can play against a human in real-time.

---

## Features

- **Neural Network-Based AI**: Uses a convolutional neural network (CNN) to evaluate board positions.  
- **CUDA-Accelerated**: If a GPU is available, training and inference use PyTorch with CUDA.  
- **Pygame GUI**: Clean chessboard rendering with piece logos and interactive play.  
- **Training Features for Stability**:  
  - Discounted value targets based on move depth  
  - Gradient clipping  
  - Scaled target values to prevent early saturation  
  - Per-epoch loss and learning rate logging  

---

## Folder Structure

```
project/
├── main.py           # Main Python file
├── chess_net.pth         # Neural network weights (created after training)
├── assets/               # Folder containing piece images
│   ├── wp.png
│   ├── wn.png
│   ├── wb.png
│   ├── wr.png
│   ├── wq.png
│   ├── wk.png
│   ├── bp.png
│   ├── bn.png
│   ├── bb.png
│   ├── br.png
│   ├── bq.png
│   └── bk.png
```

---

## Piece Images

| File | Piece |
|------|-------|
| `wp.png` | White Pawn |
| `wn.png` | White Knight |
| `wb.png` | White Bishop |
| `wr.png` | White Rook |
| `wq.png` | White Queen |
| `wk.png` | White King |
| `bp.png` | Black Pawn |
| `bn.png` | Black Knight |
| `bb.png` | Black Bishop |
| `br.png` | Black Rook |
| `bq.png` | Black Queen |
| `bk.png` | Black King |

---

## Installation

1. Clone this repository or download the files.
2. Install Python dependencies:

```bash
pip install torch torchvision torchaudio pygame chess numpy
```

3. Ensure CUDA is installed if you want GPU acceleration.

---

## Usage

1. Run the engine:

```bash
python main.py
```

2. The engine will train a neural network if no model exists.  
   - Training logs show **loss** and **learning rate** per epoch.  
   - The trained model is saved as `chess_net.pth`.

3. Play by clicking on a piece and then the target square.  
4. The AI will automatically respond with its move.

---

## Training Notes

- Training uses random self-play games.  
- Targets are **discounted by move depth**, so early moves are less weighted.  
- Gradient clipping is applied for stability.  
- Loss will **never reach zero**, which is normal for a noisy value-based network.  

---

## Recommended Improvements

- Add a **policy head** to predict move probabilities.  
- Use **MCTS** instead of pure minimax for stronger play.  
- Use **engine-guided self-play** for more reliable learning.  
- Log evaluation metrics or plot training curves for monitoring.  

---

## References

- [PyTorch](https://pytorch.org/) – Deep learning framework  
- [Python-Chess](https://python-chess.readthedocs.io/) – Chess library  
- [Pygame](https://www.pygame.org/) – Game GUI  
- Inspired by value network learning methods like AlphaZero
