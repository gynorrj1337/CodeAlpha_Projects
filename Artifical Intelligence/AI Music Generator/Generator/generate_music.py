import os
import torch
import pickle
import numpy as np
from train_model import MusicLSTM
import argparse

def load_model_and_vocab(model_path, vocab_path):
    """Load trained model and vocabulary"""
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    note_to_int = vocab_data['note_to_int']
    int_to_note = vocab_data['int_to_note']
    vocab = vocab_data['vocab']
    stats = vocab_data.get('stats', {})
    
    # Initialize model
    model = MusicLSTM(len(vocab))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, note_to_int, int_to_note, vocab, stats

def generate_music_sequence(model, note_to_int, int_to_note, start_seq=None, 
                          length=100, temperature=0.8, top_k=10):
    """Generate music sequence using trained LSTM model"""
    if start_seq is None:
        # Start with a random sequence from vocabulary
        start_seq = [np.random.choice(list(note_to_int.values())) for _ in range(10)]
    else:
        # Convert start sequence to integers if needed
        if isinstance(start_seq[0], str) or (isinstance(start_seq[0], int) and start_seq[0] > 100):
            start_seq = [note_to_int[note] for note in start_seq if note in note_to_int]
    
    generated = start_seq.copy()
    
    with torch.no_grad():
        hidden = None
        current_input = torch.tensor([start_seq], dtype=torch.long)
        
        for _ in range(length):
            output, hidden = model(current_input, hidden)
            
            # Apply temperature and top-k sampling
            output = output[0, -1] / temperature
            probabilities = torch.softmax(output, dim=0).cpu().numpy()
            
            # Top-k sampling
            top_k_indices = np.argsort(probabilities)[-top_k:]
            top_k_probs = probabilities[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)  # Renormalize
            
            next_note_idx = np.random.choice(top_k_indices, p=top_k_probs)
            generated.append(next_note_idx)
            
            # Update input for next prediction
            current_input = torch.tensor([[next_note_idx]], dtype=torch.long)
    
    # Convert back to MIDI note numbers
    generated_notes = [int_to_note[idx] for idx in generated]
    return generated_notes

def save_sequence_to_midi(sequence, filename):
    """Save note sequence to MIDI file"""
    from music21 import stream, note, tempo
    
    # Create a music21 stream
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=120))
    
    # Add notes to stream
    for note_val in sequence:
        if 0 <= note_val <= 127:  # Valid MIDI range
            n = note.Note(note_val)
            n.duration.quarterLength = 0.5  # Half note duration
            s.append(n)
    
    # Write MIDI file
    s.write('midi', fp=filename)
    print(f"MIDI file saved: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate music using trained LSTM model')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                       help='Directory containing the trained model and vocabulary')
    parser.add_argument('--output_dir', type=str, default='generated_music',
                       help='Directory to save generated music')
    parser.add_argument('--length', type=int, default=100,
                       help='Length of generated sequence')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k sampling parameter')
    parser.add_argument('--num_pieces', type=int, default=5,
                       help='Number of pieces to generate')
    
    args = parser.parse_args()
    
    # Load model and vocabulary
    model_path = os.path.join(args.model_dir, 'music_lstm_model.pth')
    vocab_path = os.path.join(args.model_dir, 'music_vocabulary.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print("Error: Model or vocabulary file not found. Please train the model first.")
        return
    
    print("Loading model and vocabulary...")
    model, note_to_int, int_to_note, vocab, stats = load_model_and_vocab(model_path, vocab_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_pieces} music pieces...")
    for i in range(args.num_pieces):
        print(f"\nGenerating piece {i+1}/{args.num_pieces}...")
        
        # Generate music sequence
        generated_sequence = generate_music_sequence(
            model, note_to_int, int_to_note, 
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Save to MIDI file
        output_path = os.path.join(args.output_dir, f'generated_piece_{i+1}.mid')
        save_sequence_to_midi(generated_sequence, output_path)
        
        print(f"  Generated {len(generated_sequence)} notes")
        print(f"  Note range: {min(generated_sequence)} - {max(generated_sequence)}")
    
    print(f"\n✅ Generated {args.num_pieces} pieces in {args.output_dir}/")

if __name__ == '__main__':
    main()