import torch
from funasr import AutoModel


class Emotion2Vec:

    def __init__(self, model_id="iic/emotion2vec_plus_base") -> None:
        """
        Available emotion2vec model variants for `model_id`:
        - "iic/emotion2vec_base"
        - "iic/emotion2vec_base_finetuned"
        - "iic/emotion2vec_plus_seed"
        - "iic/emotion2vec_plus_base" (default)
        """

        self.emotion2vec = AutoModel(
            model=model_id,
            hub="hf",  # Use hugging face
        )


    @torch.no_grad()
    def extract_emotion_embeddings(self, wav_path, return_embeddings_only=True):
        result = self.emotion2vec.generate(
            wav_path,
            output_dir="./emotion2vec_outputs",
            granularity="utterance",
            extract_embedding=True
        )[0]

        if not return_embeddings_only:
            return {
                "embeddings": result["feats"],
                "predicted_emotion": max(result["labels"]),
                "predicted_score": max(result["scores"])
            }
        
        return result["feats"]
