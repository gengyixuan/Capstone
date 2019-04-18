from data_models.combine import data_model_combine
from data_models.meta import data_model_meta
from data_models.word_embedding import data_model_word_embedding
from data_models.ranking import ranking


data_model_list = {
    'WordEmbedding': data_model_word_embedding,
    'Metadata': data_model_meta,
    'Combined': data_model_combine,
    'Ranking': ranking
}