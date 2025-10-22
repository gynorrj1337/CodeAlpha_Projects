import os
import uuid
import random
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from scipy.io import wavfile

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/generated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global status
generation_status = {
    'progress': 0,
    'status': 'Idle',
    'filename': None
}

# Drastically different music generators for each genre
class MusicGenerator:
    @staticmethod
    def classical(length=50):
        """Classical: Structured, melodic, harmonic progressions"""
        # Classical chord progressions (I-IV-V-I)
        chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]  # C-F-G-C
        melody = []
        
        for i in range(length):
            chord = chords[i % len(chords)]
            # Arpeggiate the chord with some passing notes
            if i % 4 == 0:
                melody.extend(chord)  # Play chord
            else:
                # Add melodic passing notes
                note = chord[0] + random.choice([0, 2, 4, 5, 7, 9])
                melody.append(min(84, max(48, note)))
        
        return melody[:length]

    @staticmethod
    def jazz(length=50):
        """Jazz: Complex chords, syncopation, blues notes"""
        # Jazz chords (ii-V-I progression with extensions)
        chords = [[62, 65, 69, 72], [67, 71, 74, 77], [60, 64, 67, 70]]  # Dm9-G9-Cmaj7
        melody = []
        blues_notes = [60, 63, 65, 66, 67, 70]  # Blues scale
        
        for i in range(length):
            if random.random() < 0.3:  # 30% chance for blues note
                note = random.choice(blues_notes) + random.randint(-12, 12)
            else:
                chord = chords[i % len(chords)]
                # Use chord tones with occasional extensions
                if random.random() < 0.7:
                    note = random.choice(chord)
                else:
                    note = chord[0] + random.choice([2, 4, 7, 9, 11])
            
            melody.append(min(84, max(48, note)))
            
            # Add syncopation - sometimes skip beats
            if random.random() < 0.2:
                melody.append(0)  # Rest
        
        return melody[:length]

    @staticmethod
    def blues(length=50):
        """Blues: 12-bar structure, soulful, repetitive patterns"""
        # 12-bar blues progression
        chords = [[60, 63, 67], [60, 64, 67], [60, 63, 67], [60, 63, 67],
                 [65, 69, 72], [65, 69, 72], [60, 63, 67], [60, 63, 67],
                 [67, 71, 74], [65, 69, 72], [60, 63, 67], [67, 71, 74]]
        
        melody = []
        blues_scale = [60, 63, 65, 66, 67, 70, 72]  # C Blues scale
        
        for i in range(length):
            chord_idx = (i // 4) % len(chords)  # 4 notes per chord
            chord = chords[chord_idx]
            
            # Blues style: mix of chord tones and blues scale
            if random.random() < 0.6:
                note = random.choice(blues_scale)
            else:
                note = random.choice(chord)
            
            # Add soulful bends and slides
            if random.random() < 0.3:
                melody.append(note)
                melody.append(note + random.choice([-1, 1]))  # Grace note
            
            melody.append(note)
        
        return melody[:length]

    @staticmethod
    def rock(length=50):
        """Rock: Power chords, strong rhythm, repetitive riffs"""
        # Rock power chords
        power_chords = [[60, 64], [62, 65], [64, 67], [65, 69], [67, 71]]
        melody = []
        
        # Create a rock riff
        riff = [60, 60, 60, 64, 64, 64, 65, 65, 67, 67]
        
        for i in range(length):
            if i % 10 == 0:  # Change riff every 10 notes
                base_note = random.choice([48, 50, 52, 53, 55, 57])  # Lower register
                riff = [base_note, base_note, base_note, 
                       base_note + 4, base_note + 4, base_note + 4,
                       base_note + 5, base_note + 5,
                       base_note + 7, base_note + 7]
            
            note = riff[i % len(riff)]
            melody.append(note)
            
            # Double some notes for power
            if random.random() < 0.4:
                melody.append(note)
        
        return melody[:length]

    @staticmethod
    def pop(length=50):
        """Pop: Catchy, repetitive, simple harmonies"""
        # Common pop progression
        chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [62, 65, 69]]  # C-F-Am-G
        melody = []
        
        # Create a hook/melody
        hook = [60, 64, 67, 64, 60, 65, 67, 65]
        
        for i in range(length):
            chord = chords[i % len(chords)]
            
            if i % 8 == 0:  # Repeat hook every 8 notes
                note = hook[i % len(hook)]
            else:
                # Simple stepwise motion
                if melody and random.random() < 0.7:
                    last_note = melody[-1]
                    note = last_note + random.choice([-2, -1, 1, 2])
                else:
                    note = random.choice(chord)
            
            melody.append(min(76, max(60, note)))  # Stay in comfortable pop range
        
        return melody[:length]

    @staticmethod
    def electronic(length=50):
        """Electronic: Repetitive, synthetic, arpeggiated"""
        # Arpeggio patterns
        arpeggios = [
            [60, 64, 67, 72],  # C Major arpeggio
            [60, 63, 67, 70],  # C Minor arpeggio
            [60, 65, 69, 72],  # C Add9
            [60, 67, 72, 79]   # C 5ths
        ]
        
        melody = []
        current_arpeggio = random.choice(arpeggios)
        pattern_length = random.choice([4, 8, 16])
        
        for i in range(length):
            if i % pattern_length == 0:
                current_arpeggio = random.choice(arpeggios)
                # Sometimes transpose
                if random.random() < 0.3:
                    transpose = random.choice([0, 5, 7, 12])
                    current_arpeggio = [n + transpose for n in current_arpeggio]
            
            # Arpeggiate
            note = current_arpeggio[i % len(current_arpeggio)]
            
            # Electronic style: lots of repetition
            if random.random() < 0.6:
                melody.append(note)
                melody.append(note)  # Double the note
            
            melody.append(note)
        
        return melody[:length]

    @staticmethod
    def mixed(genres, length=50):
        """Mixed: Combine characteristics from multiple genres"""
        if not genres:
            return MusicGenerator.classical(length)
        
        # Pick a primary genre and add influences from others
        primary = random.choice(genres)
        secondary = [g for g in genres if g != primary]
        
        # Generate base from primary genre
        if primary == 'classical':
            melody = MusicGenerator.classical(length)
        elif primary == 'jazz':
            melody = MusicGenerator.jazz(length)
        elif primary == 'blues':
            melody = MusicGenerator.blues(length)
        elif primary == 'rock':
            melody = MusicGenerator.rock(length)
        elif primary == 'pop':
            melody = MusicGenerator.pop(length)
        elif primary == 'electronic':
            melody = MusicGenerator.electronic(length)
        else:
            melody = MusicGenerator.classical(length)
        
        # Add influences from secondary genres
        for genre in secondary:
            influence_length = length // len(secondary)
            influence_start = random.randint(0, length - influence_length)
            
            if genre == 'jazz':
                # Add some blues notes
                for i in range(influence_start, min(length, influence_start + influence_length)):
                    if random.random() < 0.3:
                        melody[i] = random.choice([60, 63, 65, 66, 67, 70])
            
            elif genre == 'blues':
                # Add soulful repetitions
                for i in range(influence_start, min(length, influence_start + influence_length), 3):
                    if i + 1 < length:
                        melody[i + 1] = melody[i]  # Repeat note
            
            elif genre == 'rock':
                # Add power chord elements
                for i in range(influence_start, min(length, influence_start + influence_length)):
                    if random.random() < 0.2:
                        melody[i] = random.choice([48, 50, 52, 53, 55])  # Lower notes
            
            elif genre == 'electronic':
                # Add arpeggio patterns
                for i in range(influence_start, min(length, influence_start + influence_length), 4):
                    if i + 3 < length:
                        melody[i + 1] = melody[i] + 4
                        melody[i + 2] = melody[i] + 7
                        melody[i + 3] = melody[i] + 12
        
        return melody

# Audio generation functions
def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5, wave_type='sine'):
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if wave_type == 'sine':
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'sawtooth':
        wave = amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
    elif wave_type == 'triangle':
        wave = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1)
    else:
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return wave

def note_to_frequency(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def apply_envelope(signal_array, attack=0.1, decay=0.1, sustain=0.7, release=0.2, sample_rate=44100):
    total_samples = len(signal_array)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    
    if sustain_samples < 0:
        sustain_samples = 0
        release_samples = total_samples - attack_samples - decay_samples
    
    envelope = np.ones(total_samples)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if decay_samples > 0:
        start_decay = attack_samples
        end_decay = start_decay + decay_samples
        if end_decay <= total_samples:
            envelope[start_decay:end_decay] = np.linspace(1, sustain, decay_samples)
    
    if sustain_samples > 0:
        start_sustain = attack_samples + decay_samples
        end_sustain = start_sustain + sustain_samples
        if end_sustain <= total_samples:
            envelope[start_sustain:end_sustain] = sustain
    
    if release_samples > 0:
        start_release = total_samples - release_samples
        if start_release >= 0:
            envelope[start_release:] = np.linspace(envelope[start_release], 0, release_samples)
    
    return signal_array * envelope

def get_genre_waveform(genre):
    """Different waveforms for different genres"""
    waveforms = {
        'classical': 'sine',
        'jazz': 'sawtooth',
        'blues': 'triangle',
        'rock': 'square',
        'pop': 'sine',
        'electronic': 'sawtooth'
    }
    return waveforms.get(genre, 'sine')

def get_genre_tempo(genre):
    """Different tempos for different genres"""
    tempos = {
        'classical': 0.4,
        'jazz': 0.3,
        'blues': 0.5,
        'rock': 0.2,
        'pop': 0.3,
        'electronic': 0.15
    }
    return tempos.get(genre, 0.3)

def create_audio_from_melody(melody, genres, filename, sample_rate=44100):
    try:
        # Use characteristics from primary genre
        primary_genre = genres[0] if genres else 'classical'
        note_duration = get_genre_tempo(primary_genre)
        waveform = get_genre_waveform(primary_genre)
        amplitude = 0.3
        
        audio_segments = []
        for note in melody:
            if note == 0:  # Rest
                rest_audio = np.zeros(int(note_duration * sample_rate))
                audio_segments.append(rest_audio)
            elif 0 <= note <= 127:
                frequency = note_to_frequency(note)
                note_audio = generate_tone(frequency, note_duration, sample_rate, amplitude, waveform)
                note_audio = apply_envelope(note_audio, attack=0.05, decay=0.1, sustain=0.7, release=0.15)
                audio_segments.append(note_audio)
            else:
                rest_audio = np.zeros(int(note_duration * sample_rate))
                audio_segments.append(rest_audio)
        
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
        else:
            full_audio = np.zeros(int(2 * sample_rate))
        
        silence_duration = 0.1
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        full_audio = np.concatenate([silence, full_audio, silence])
        
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.8
        
        full_audio = (full_audio * 32767).astype(np.int16)
        
        wavfile.write(filename, sample_rate, full_audio)
        
        print(f"Audio file created: {filename}")
        return True
        
    except Exception as e:
        print(f"Error creating audio: {e}")
        return False

def convert_wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using pydub"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        print(f"Converted to MP3: {mp3_path}")
        return True
    except Exception as e:
        print(f"MP3 conversion failed: {e}")
        return False

def generate_drastically_different_music(genres):
    """Generate music that's drastically different for each genre combination"""
    global generation_status
    
    try:
        generation_status['progress'] = 20
        generation_status['status'] = f'Creating {", ".join(genres)} style music...'
        
        # Vary length for more variety
        length = random.randint(40, 80)
        
        # Generate melody based on genres
        if len(genres) == 1:
            # Single genre - use specific generator
            genre = genres[0]
            if genre == 'classical':
                melody = MusicGenerator.classical(length)
            elif genre == 'jazz':
                melody = MusicGenerator.jazz(length)
            elif genre == 'blues':
                melody = MusicGenerator.blues(length)
            elif genre == 'rock':
                melody = MusicGenerator.rock(length)
            elif genre == 'pop':
                melody = MusicGenerator.pop(length)
            elif genre == 'electronic':
                melody = MusicGenerator.electronic(length)
            else:
                melody = MusicGenerator.classical(length)
        else:
            # Multiple genres - use mixed generator
            melody = MusicGenerator.mixed(genres, length)
        
        generation_status['progress'] = 60
        generation_status['status'] = 'Rendering audio...'
        
        # Create unique filename with genre info
        genre_str = "_".join(sorted(genres))[:20]
        unique_id = str(uuid.uuid4())[:6]
        wav_filename = f"{genre_str}_{unique_id}.wav"
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
        
        # Create audio with genre-specific characteristics
        if not create_audio_from_melody(melody, genres, wav_path):
            raise Exception("Failed to create audio file")
        
        generation_status['progress'] = 80
        generation_status['status'] = 'Finalizing...'
        
        # Convert to MP3
        mp3_filename = wav_filename.replace('.wav', '.mp3')
        mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
        
        if convert_wav_to_mp3(wav_path, mp3_path):
            final_filename = mp3_filename
            final_path = mp3_path
        else:
            final_filename = wav_filename
            final_path = wav_path
        
        generation_status['progress'] = 100
        generation_status['status'] = 'Music Ready!'
        generation_status['filename'] = final_filename
        
        print(f"🎵 Generated {genres} music: {final_filename}")
        print(f"   Notes: {len(melody)}, Range: {min(melody)}-{max(melody)}")
        
        return final_path, final_filename
        
    except Exception as e:
        generation_status['status'] = f'Generation failed: {str(e)}'
        generation_status['progress'] = 0
        raise

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    global generation_status
    
    try:
        genres = request.form.getlist('genres')
        if not genres:
            return jsonify({'status': 'error', 'message': 'No genres selected'})
        
        print(f"🎹 Generating DRAMATICALLY DIFFERENT music for: {genres}")
        
        # Reset status
        generation_status = {
            'progress': 0,
            'status': f'Creating unique {", ".join(genres)} composition...',
            'filename': None
        }
        
        # Generate drastically different music
        audio_path, filename = generate_drastically_different_music(genres)
        
        print(f"✅ Successfully generated: {filename}")
        
        return jsonify({
            'status': 'success', 
            'message': f'Created unique {", ".join(genres)} music!',
            'filename': filename
        })
        
    except Exception as e:
        print(f"❌ Error in generate route: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/status')
def get_status():
    return jsonify(generation_status)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return "File not found", 404
            
    except Exception as e:
        return f"Download error: {str(e)}", 500

@app.route('/generated/<filename>')
def serve_generated_file(filename):
    try:
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            if filename.lower().endswith('.mp3'):
                mimetype = 'audio/mpeg'
            elif filename.lower().endswith('.wav'):
                mimetype = 'audio/wav'
            else:
                mimetype = 'application/octet-stream'
            
            return send_file(file_path, mimetype=mimetype)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    print("🎵 DRAMATICALLY DIFFERENT Music Generator")
    print("=" * 60)
    print("🎹 Each genre combination creates COMPLETELY different music:")
    print("   • Classical: Structured melodies, chord progressions")
    print("   • Jazz: Complex chords, blues notes, syncopation") 
    print("   • Blues: 12-bar structure, soulful patterns")
    print("   • Rock: Power chords, strong riffs, repetition")
    print("   • Pop: Catchy hooks, simple harmonies")
    print("   • Electronic: Arpeggios, synthetic patterns")
    print("   • Mixed: Unique combinations of above")
    print("")
    print(f"📁 Output: {app.config['UPLOAD_FOLDER']}")
    print("🌐 Server: http://localhost:5000")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5000)