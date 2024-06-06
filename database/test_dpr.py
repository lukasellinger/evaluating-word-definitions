from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from models.evidence_selection_model import EvidenceSelectionModel

tokenizer = AutoTokenizer.from_pretrained('lukasellinger/evidence_selection_model-v1')
model = AutoModel.from_pretrained('lukasellinger/evidence_selection_model-v1', trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
selection_model = EvidenceSelectionModel(model)


query = selection_model(**tokenizer('Hi wie geht es dir.', return_tensors='pt'))
document1 = selection_model(**tokenizer('Das geht hier wirklich ab.', return_tensors='pt'))
document2 = selection_model(**tokenizer('Das geht hier wirklich ab Hi wie gehts.', return_tensors='pt'))
document = selection_model(**tokenizer(['Das geht hier wirklich ab Hie wie gehts.', 'Das geht hier wirklich ab'], padding=True, return_tensors='pt'))



print(F.cosine_similarity(query, document1, dim=2))
print(F.cosine_similarity(query, document2, dim=2))
print(F.cosine_similarity(query, document, dim=2))



