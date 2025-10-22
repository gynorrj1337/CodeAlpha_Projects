import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import music21 as m21
import glob
from tqdm import tqdm
import argparse

class MIDIDataset(Dataset):
    def __init__(self, sequences, sequence_length=50):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.valid_sequences = []
        
        for seq in sequences:
            if len(seq) > sequence_length:
                for i in range(len(seq) - sequence_length):
                    self.valid_sequences.append((seq, i))
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        seq, start_idx = self.valid_sequences[idx]
        input_seq = seq[start_idx:start_idx + self.sequence_length]
        target_seq = seq[start_idx + 1:start_idx + self.sequence_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(MusicLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

def parse_midi_file(midi_path):
    """Parse a single MIDI file and extract note sequences"""
    try:
        print(f"  Parsing: {os.path.basename(midi_path)}")
        midi = m21.converter.parse(midi_path)
        
        notes = []
        for element in midi.flatten().notes:
            if isinstance(element, m21.note.Note):
                notes.append(element.pitch.midi)
            elif isinstance(element, m21.chord.Chord):
                # Use the root note of the chord
                notes.append(element.root().midi)
        
        return notes
    except Exception as e:
        print(f"  Error parsing {midi_path}: {e}")
        return None

def load_and_preprocess_data(dataset_path, genres=None, max_files_per_genre=50):
    """Load MIDI files and preprocess them into sequences"""
    if genres is None:
        genres = ['classical', 'jazz', 'blues', 'rock', 'pop', 'electronic']
    
    all_sequences = []
    genre_stats = {}
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.exists(genre_path):
            print(f"Warning: Genre directory not found: {genre_path}")
            continue
            
        print(f"Processing {genre}...")
        midi_files = glob.glob(os.path.join(genre_path, "*.mid")) + \
                    glob.glob(os.path.join(genre_path, "*.midi"))
        
        print(f"  Found {len(midi_files)} MIDI files")
        
        genre_sequences = []
        files_processed = 0
        
        for midi_file in tqdm(midi_files[:max_files_per_genre], desc=f"  {genre}"):
            notes = parse_midi_file(midi_file)
            if notes and len(notes) >= 20:  # Only use sequences with reasonable length
                genre_sequences.append(notes)
                files_processed += 1
        
        if genre_sequences:
            all_sequences.extend(genre_sequences)
            total_notes = sum(len(seq) for seq in genre_sequences)
            genre_stats[genre] = {
                'files_processed': files_processed,
                'total_sequences': len(genre_sequences),
                'total_notes': total_notes,
                'avg_sequence_length': total_notes / len(genre_sequences)
            }
            print(f"  ✓ Processed {files_processed} files, {len(genre_sequences)} sequences")
        else:
            print(f"  ✗ No valid sequences found for {genre}")
    
    return all_sequences, genre_stats

def create_vocabulary(sequences):
    """Create vocabulary from all sequences"""
    all_notes = []
    for seq in sequences:
        all_notes.extend(seq)
    
    vocab = sorted(set(all_notes))
    note_to_int = {note: i for i, note in enumerate(vocab)}
    int_to_note = {i: note for note, i in note_to_int.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Note range: {min(vocab)} - {max(vocab)}")
    
    return vocab, note_to_int, int_to_note

def convert_sequences_to_int(sequences, note_to_int):
    """Convert note sequences to integer sequences"""
    int_sequences = []
    for seq in sequences:
        int_seq = [note_to_int[note] for note in seq]
        int_sequences.append(int_seq)
    return int_sequences

def train_model(int_sequences, vocab_size, sequence_length=50, num_epochs=100, batch_size=64):
    """Train the LSTM model"""
    # Create dataset and dataloader
    dataset = MIDIDataset(int_sequences, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training on {len(dataset)} sequences")
    print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
    
    # Initialize model
    model = MusicLSTM(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model, losses

def save_model_and_vocab(model, note_to_int, int_to_note, vocab, stats, save_dir='saved_models'):
    """Save the trained model and vocabulary"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'music_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save vocabulary and metadata
    vocab_path = os.path.join(save_dir, 'music_vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({
            'note_to_int': note_to_int,
            'int_to_note': int_to_note,
            'vocab': vocab,
            'stats': stats
        }, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return model_path, vocab_path

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model on MIDI data')
    parser.add_argument('--dataset_path', type=str, default='music_dataset', 
                       help='Path to the music dataset directory')
    parser.add_argument('--genres', nargs='+', 
                       default=['classical', 'jazz', 'blues', 'rock', 'pop', 'electronic'],
                       help='Genres to include in training')
    parser.add_argument('--max_files', type=int, default=50,
                       help='Maximum number of files to process per genre')
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='Sequence length for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    print("🎵 MIDI Data Preprocessing and Model Training")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing MIDI data...")
    sequences, stats = load_and_preprocess_data(
        args.dataset_path, 
        args.genres, 
        args.max_files
    )
    
    if not sequences:
        print("Error: No sequences found. Please check your dataset path.")
        return
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 30)
    for genre, stat in stats.items():
        print(f"{genre:12} | Files: {stat['files_processed']:3d} | "
              f"Sequences: {stat['total_sequences']:4d} | "
              f"Notes: {stat['total_notes']:6d} | "
              f"Avg Len: {stat['avg_sequence_length']:5.1f}")
    
    total_sequences = sum(stat['total_sequences'] for stat in stats.values())
    total_notes = sum(stat['total_notes'] for stat in stats.values())
    print(f"\nTotal: {total_sequences} sequences, {total_notes} notes")
    
    # Step 2: Create vocabulary
    print("\nStep 2: Creating vocabulary...")
    vocab, note_to_int, int_to_note = create_vocabulary(sequences)
    
    # Step 3: Convert sequences to integers
    print("\nStep 3: Converting sequences to integer representation...")
    int_sequences = convert_sequences_to_int(sequences, note_to_int)
    
    # Step 4: Train model
    print("\nStep 4: Training LSTM model...")
    model, losses = train_model(
        int_sequences, 
        len(vocab), 
        sequence_length=args.sequence_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 5: Save model and vocabulary
    print("\nStep 5: Saving model and vocabulary...")
    model_path, vocab_path = save_model_and_vocab(
        model, note_to_int, int_to_note, vocab, stats, args.save_dir
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Model saved in: {args.save_dir}")

if __name__ == '__main__':
    main()