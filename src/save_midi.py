from music21 import instrument, note, stream, chord, duration
import os


def save_notes_and_durations(generated, filepath):
    midi_stream = stream.Stream()
    for gen_notes, gen_durations in zip(*generated):
        gen_notes = gen_notes.split('.')[:-1]
        # chord
        if len(gen_notes) > 1:
            cur_chord = []
            for cur_note in gen_notes:
                new_note = note.Note(int(cur_note))
                new_note.duration = duration.Duration(gen_durations)
                new_note.soredInstrument = instrument.Percussion()
                cur_chord.append(new_note)
            midi_stream.append(chord.Chord(cur_chord))
        elif gen_notes != []:
            if gen_notes[0] == 'del':
                new_note = note.Rest()
                new_note.duration = duration.Duration(type=1)
                new_note.storedInstrument = instrument.Percussion()
                midi_stream.append(new_note)
            else:
                new_note = note.Note(int(gen_notes[0]))
                new_note.duration = duration.Duration(gen_durations)
                new_note.soredInstrument = instrument.Percussion()
                midi_stream.append(new_note)
        else:
            new_note = note.Rest()
            new_note.duration = duration.Duration(gen_durations)
            new_note.storedInstrument = instrument.Percussion()
            midi_stream.append(new_note)
            
    midi_stream = midi_stream.chordify()
    midi_stream.write('midi', fp=os.path.join(filepath + '.mid'))


def save_notes(generated, filepath):
    midi_stream = stream.Stream()
    for gen_notes in generated:
        gen_notes = gen_notes.split('.')[:-1]
        # chord
        if len(gen_notes) > 1:
            cur_chord = []
            for cur_note in gen_notes:
                new_note = note.Note(int(cur_note))
                new_note.duration = duration.Duration(type='16th')
                new_note.soredInstrument = instrument.Percussion()
                cur_chord.append(new_note)
            midi_stream.append(chord.Chord(cur_chord))
        elif gen_notes != []:
            if gen_notes[0] == 'del':
                new_note = note.Rest()
                new_note.duration = duration.Duration(type=1)
                new_note.storedInstrument = instrument.Percussion()
                midi_stream.append(new_note)
            else:
                new_note = note.Note(int(gen_notes[0]))
                new_note.duration = duration.Duration(type='16th')
                new_note.soredInstrument = instrument.Percussion()
                midi_stream.append(new_note)
        else:
            new_note = note.Rest()
            new_note.duration = duration.Duration(type='16th')
            new_note.storedInstrument = instrument.Percussion()
            midi_stream.append(new_note)
            
    midi_stream = midi_stream.chordify()
    midi_stream.write('midi', fp=os.path.join(filepath + '.mid'))