import time

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# Load environment variables from the .env file
load_dotenv()


# Define one source used to support the generated answer
class Source(BaseModel):

    chunk_id: int = Field(
        description="The ID of a retrieved chunk that directly supports the answer."
    ) 


# Define structure of the final response
class AnswerResponse(BaseModel):

    question: str = Field(
        description="The user's original question."
    )

    answer: str = Field(
        description="The answer generated only from the retrieved document context."
    )

    sources: list[Source] = Field(
        description="The retrieved chunks that directly support the answer."
    )


class LLM_Model:

    def __init__(self):

        # Initialize the Gemini client
        #
        # The client automatically looks for:
        # GEMINI_API_KEY
        #
        # in the environment
        self.client = genai.Client()

    
        self.model_names = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash"
        ]

    def build_context(self, retrieved_chunks):

        # Create an empty list that will hold each formatted retrieved chunk
        context_parts = []

        # Go through the reranked chunks returned by the CrossEncoder
        for result in retrieved_chunks:

            #chunk ID
            chunk_id = result["chunk_id"]

            #document text
            text = result["text"]

            chunk_context = (
                f"Chunk ID: {chunk_id}\n"
                f"Content:\n{text}"
            )

            context_parts.append(chunk_context)

        #Separate chunks so their boundaries are clear in the prompt.
        context = "\n\n---\n\n".join(context_parts)

        return context


    def generate_answer(self, query_text, retrieved_chunks):

        # Build document context from the retreived chunks 
        context = self.build_context(retrieved_chunks)


        # Instructions for how the model behaves -it can answer only from retrieved evidence
        # rather than relying on outside knowledge.
        system_instruction = """
You are a technical document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not infer facts that are not supported by the provided context.
3. If the provided context does not contain enough information to answer
   the question, clearly state that the available context is insufficient.
4. Only include a chunk ID in the sources if that chunk directly supports
   the answer.
5. Do not cite a chunk simply because it was retrieved.
6. If multiple chunks support the answer, synthesize the information
   across those chunks.
7. Keep the answer concise and focused on the user's question.
"""


        # Send the query and retrieved evidence to Gemini.
        prompt = f"""
USER QUESTION:
{query_text}

DOCUMENT CONTEXT:
{context}
"""


        max_retries = 3

        # Store the last error in case all models fail
        last_error = None


        #try each Gemini model 
        for model_name in self.model_names:

            print(f"Trying {model_name}...")

            for attempt in range(max_retries):

                try:

                    #grounded question + context sent to Gemini.
                 
                    interaction = self.client.interactions.create(
                        model=model_name,
                        system_instruction=system_instruction,
                        input=prompt,
                        response_format={                  # response_format tells Gemini to return JSON to match  answer response schema                                                           
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": AnswerResponse.model_json_schema()
                        },
                    )


                    # get the model name and the response it generated

                     # Validate Gemini's returned JSON using Pydantic to follow the structured response output expected
                    validated_response = (
                        AnswerResponse.model_validate_json(
                            interaction.output_text
                        )
                    )

                    print(
                        f"Response generated with {model_name}"
                    )
                    return validated_response, model_name


                # ERROR HANDLING (GEMINI):  API may wrap HTTP errors using internal SDK exception classes.
                # Inspect the error information for known API conditions.
    
                # IF Unknown errors ->  raise again below so they are not silently hidden.
                except Exception as error:

                    #save the error in case every model fails & identify error using text
                    last_error = error

                    error_text = str(error)


                    # Rate Limit - HTTP 429 indicates that the current
                    # model has reached a rate or quota limit  = move to next model, no retry
                    if (
                        "429" in error_text
                        or "Rate limit exceeded" in error_text
                        or "rate limit exceeded" in error_text
                        or "too_many_requests" in error_text
                    ):

                        print(
                            f"{model_name} rate limit reached. "
                            "Trying fallback model..."
                        )

                        break


                    # Temp server error - HTTP 503 indicates temporary service
                    # unavailability - retry may work
                    if (
                        "503" in error_text
                        or "Service Unavailable" in error_text
                        or "service_unavailable" in error_text
                        or "temporarily unavailable" in error_text
                    ):

                        # retry if attempts available
                        if attempt < max_retries - 1:

                            # Exponential backoff: 1st retry 2 sec, 2nd 4 sec
                            wait_time = 2 ** (attempt + 1)

                            print(
                                f"{model_name} temporarily unavailable. "
                                f"Retrying in {wait_time} seconds..."
                            )

                            time.sleep(wait_time)

                            continue


                        # next model after failed retry
                        print(
                            f"{model_name} remained unavailable. "
                            "Trying fallback model..."
                        )

                        break


                    # do not silently hide unexpected errors.
                    raise


        # If execution reaches this point, every configured
        # Gemini model either encountered a rate/quota limit
        # or remained unavailable after retries.
        raise RuntimeError(
            "All configured Gemini models failed."
        ) from last_error