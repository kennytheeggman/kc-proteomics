import torch
from torch import nn

# embedding FFNs, taken largely from dreams code
class FeedForward(nn.Module):

    # hidden_dims is [size of hidden layer 1, size of hidden layer 2, etc]
    def __init__(self, input_dim, output_dim, hidden_dims, normalizer=nn.ReLU, bias=True, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.ModuleList([])
        self.ff.append(nn.Linear(input_dim, hidden_dims[0], bias=bias))
        self.ff.append(normalizer())
        self.ff.append(nn.Dropout(p=dropout))
        for i in range(len(hidden_dims)-1):
            self.ff.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=bias))
            self.ff.append(normalizer())
            self.ff.append(nn.Dropout(p=dropout))
        self.ff.append(nn.Linear(hidden_dims[-1], output_dim, bias=bias))
        self.linear_stack = nn.Sequential(*self.linear_stack)
    
    def forward(self, x):
        x = nn.flatten(x)
        return self.linear_stack(x)


# spectra encoder
class SpectraEncoder(nn.Module):
    
    def __init__(self):
        return


# decoder into AA probability
class ProbabilityDecoder(nn.Module):

    def __init__(self):
        return


# decoder into spectra
class SpectraDecoder(nn.Module):

    def __init__(self):
        return




# make the classes?

# very basic, dont need to write own transformer or anything?
# then another file to train maybe
# and another file to run

# kenny pseudocode below

# What each segment (module) takes on or outputs

# #### Spectrum Vectorizer

# - Goal is to go from raw spectra (m/z values) to vectors ($1\times n$ tensors)

# ```python
# class SpectrumVectorizer(): # not a module
# 	def forward(self, m_z, i):
# 		mass_tensor = fourierEncode(m_z)
# 		intensity_tensor = fourierEncode(i)
# 		# enforce dim of d_vec
# 		return mass_tensor, intensity_tensor
# ```

# #### Vector Embedder

# - Goal is to go from encoded spectrum data (and precursor) to semantic embedding (that directly represents the amino acid sequence)

# ```python
# class VectorEmbedder(nn.Module):
# 	def __init__(self, d_model, d_vec, num_spec, num_prec):
# 		# expand (project up) along vec dimension to model dimension
# 		self.vector_stack = nn.Sequential(
# 			nn.Linear(d_vec, ...), ..., nn.Linear(..., d_model)
# 		)
# 		# number of layers, specify model dimensions and number of features
# 		spectrum_layer = nn.TransformerEncoderLayer(d_model, num_features = 40)
# 		self.spectrum_stack = nn.TransformerEncoder(spectrum_layer, num_spec)
# 		# number of layers, specify model dimensions and number of features
# 		precursor_layer = nn.TransformerEncoderLayer(d_model, num_features = 80)
# 		self.precursor_stack = nn.TransformerEncoder(precursor_layer, num_prec)
		
# 		# compress (project down) along feature dimension from 80 back down to 40
# 		self.embed_stack = nn.Sequential(
# 			nn.Linear(80, ...), ..., nn.Linear(..., 40)
# 		)
		
# 	def forward(self, spectrum, precursor):
# 		projected_spectrum = self.vector_stack(spectrum)
# 		projected_precursor = self.vector_stack(precursor)
# 		transformed_spectrum = self.spectrum_stack(projected_spectrum)
# 		combined = interleave(transformed_spectrum, projected_precursor)
# 		transformed_combined = self.precursor_stack(combined)
# 		embedded = self.embed_stack(transformed_combined)
# 		return embedded
# ```

# #### Embed Sequencer

# - Goal is to go from embedded data to AA probability matrix (and subsequently decoded)

# ```python
# class EmbedSequencer(nn.Module):
# 	def __init__(self, d_model):
# 		self.sequence_stack = nn.Sequential(
# 			nn.Linear(d_model, ...), ..., nn.Linear(..., 27)
# 		)
		
# 	def forward(self, embedding):
# 		raw_matrix = self.sequence_stack(embedding)
# 		prob_matrix = nn.Sofmax(raw_matrix)
# 		return prob_matrix
# ```

# #### Embed Spectralizer

# - Goal is to go from embedded data back to spectrum data

# ```python
# class EmbedSpectralizer(nn.Module):
# 	def __init__(self, d_model, num_layers):
# 		# why encoder? causality is probably not important in this case
# 		layer = nn.TransformerEncoderLayer(d_model, 40)
# 		self.spectral_stack = nn.TransformerEncoder(layer, num_layers)
		
# 		self.project_stack = nn.Sequential(
# 			nn.Linear(40, ...), ..., nn.Linear(..., 2)
# 		)
		
# 	def forward(self, embedding):
# 		attended_spectrum = self.spectral_stack(embedding)
# 		predicted_spectrum = self.project_stack(attended_spectrum)
# 		return predicted_spectrum
# ```

# #### Loss Functions

# ```python
# # AA sequence to ground truth loss
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(prob_matrix, truth, mat_length, targ_length)
# loss.backward()

# # Predicted spectrum to real spectrum loss
# mse_loss = nn.MSELoss()
# loss = mse_loss(pred, x)
# loss.backward()
# ```
