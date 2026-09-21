from sentence_transformers import SentenceTransformer

class Embedding_Model_QWEN(): 
    def __init__(self): #constructor so that we can always have something that has to be called
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B"
        )
        # self.client = InferenceClient(
        #         provider="deepinfra",     # Hugging Face automatically selects an available inference provider
        #         api_key=os.environ["HF_TOKEN"],
        # )

    def embed_data(self, chunk_text):
        embeddings = self.model.encode( 
            chunk_text )
        return embeddings

    def embed_query(self, query_text):

    # Qwen3-Embedding is instruction-aware.
    #
    # The "query" prompt tells the embedding model that
    # this text represents a retrieval query rather than
    # document content.
    #
    # Document chunks are embedded normally, while the
    # query receives the retrieval instruction.
        embeddings = self.model.encode(
            [query_text],
            prompt_name="query"
        )

        return embeddings



"""
#WHERRRE??????????????????

embedder = Embedding_Model_QWEN() # create the object

#embed the data
embedded_data = embedder.embed_data(chunk_texts)

#embed the query
embedded_query = embedder.embed_query(query_text) #embed the query outputs a numpy array

print(type(embedded_data))
print(embedded_data.shape)

print(type(embedded_query))
print(embedded_query.shape)


"""