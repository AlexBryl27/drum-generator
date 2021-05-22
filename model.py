from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Reshape
from tensorflow.keras.layers import RepeatVector, Permute, Concatenate
from tensorflow.keras.layers import Multiply, Lambda
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam


class NotesRNN:

    def __init__(
        self,
        n_notes,
        n_durations=None,
        emb_size=100,
        n_units=256,
        n_layers=2,
        use_dropout=True,
        use_attention=True,
        use_durations=False
        ):
        self.n_notes = n_notes
        self.n_durations = n_durations
        self.emb_size = emb_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        self.use_durations = use_durations
        self._build()

    def _rnn_layer(self, x, return_seq=True):
        x = LSTM(self.n_units, return_sequences=return_seq)(x)
        if self.use_dropout:
            x = Dropout(0.3)(x)
        return x

    def _build(self):
        
        notes_in = Input(shape=(None,))
        x = Embedding(self.n_notes, self.emb_size)(notes_in)
        if self.use_durations:
            durations_in = Input(shape=(None,))
            x_dur = Embedding(self.n_durations, self.emb_size)(durations_in)
            x = Concatenate()([x, x_dur])
        
        for _ in range(self.n_layers - 1):
            x = self._rnn_layer(x)

        if self.use_attention:
            x = self._rnn_layer(x)

            e = Dense(1, activation='tanh')(x)
            e = Reshape([-1])(e)

            alpha = Activation('softmax')(e)

            c = Permute([2,1])(RepeatVector(self.n_units)(alpha))
            c = Multiply()([x, c])
            c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(self.n_units,))(c)
        else:
            c = _rnn_layer(x, False)

        notes_out = Dense(self.n_notes, activation='softmax')(c)
        if self.use_durations:
            durations_out = Dense(self.n_durations, activation='softmax')(c)
            self.model = Model([notes_in, durations_in], [notes_out, durations_out])
            if self.use_attention:
                self.att_model = Model([notes_in, durations_in], alpha)
        else:
            self.model = Model(notes_in, notes_out)
            if self.use_attention:
                self.att_model = Model(notes_in, alpha)
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))