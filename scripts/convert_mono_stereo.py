import wave
import numpy as np

def stereo_to_mono(input_file: str, output_file: str):
    """
    Convert a stereo .wav file to mono by averaging the two channels.

    Args:
        input_file (str): Path to the input stereo .wav file.
        output_file (str): Path for the output mono .wav file.
    """
    with wave.open(input_file, 'rb') as wav_in:
        n_channels, sampwidth, framerate, n_frames, comptype, compname = wav_in.getparams()
        
        if n_channels != 2:
            raise ValueError("Input file must be stereo (2 channels)")
        
        frames = wav_in.readframes(n_frames)
        
        # Map sample width to numpy dtype
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ValueError("Unsupported sample width: {}".format(sampwidth))
        
        # Load data into numpy array and reshape (n_frames, 2)
        data = np.frombuffer(frames, dtype=dtype)
        data = data.reshape(-1, 2)
        
        # Average the two channels and round to nearest integer
        mono = np.round(data.mean(axis=1)).astype(dtype)
    
    # Write the mono data to the output file
    with wave.open(output_file, 'wb') as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(framerate)
        wav_out.writeframes(mono.tobytes())

# Example usage:
if __name__ == "__main__":
    input_wav = "input_stereo.wav"
    output_wav = "output_mono.wav"
    try:
        stereo_to_mono(input_wav, output_wav)
        print(f"Converted '{input_wav}' to mono and saved as '{output_wav}'.")
    except Exception as e:
        print(f"Error: {e}")