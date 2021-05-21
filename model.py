from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Reshape
from tensorflow.keras.layers import RepeatVector, Permute
from tensorflow.keras.layers import Multiply, Lambda
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam


class NotesRNN:

    def __init__(
        self,
        n_notes,
        emb_size=100,
        n_units=256,
        n_layers=2,
        use_dropout=True,
        use_attention=True
        ):
        self.n_notes = n_notes
        self.emb_size = emb_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        self._build()

    def _rnn_layer(self, x):
        x = LSTM(self.n_units, return_sequences=True)(x)
        if self.use_dropout:
            x = Dropout(0.3)(x)
        return x

    def _build(self):
        notes_in = Input(shape=(None,))
        x = Embedding(self.n_notes, self.emb_size)(notes_in)
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
            c = LSTM(self.n_units, return_sequences=False)(x)

        notes_out = Dense(self.n_notes, activation='softmax')(c)
        
        self.model = Model(notes_in, notes_out)
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))