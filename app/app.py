from datasets import load_dataset
from flask import Flask, jsonify, render_template, request

from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import Pipeline
from pipeline_module.sentence_connector import (ColonSentenceConnector,
                                                PhiSentenceConnector)
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.translator import OpusMTTranslator

app = Flask(__name__)

# Define datasets
datasets = {
    'german_dpr-claim_verification': {
        'dataset': load_dataset('lukasellinger/german_dpr-claim_verification', split='test'),
        'lang': 'de'
    },
    'german_wiktionary-claim_verification-mini': {
        'dataset': load_dataset('lukasellinger/german_wiktionary-claim_verification-mini',
                                split='test'),
        'lang': 'de'
    },
    'squad-claim_verification': {
        'dataset': load_dataset('lukasellinger/squad-claim_verification', split='test'),
        'lang': 'en'
    }
}


@app.route('/api/load_pipeline', methods=['POST'])
def load_pipeline():
    global pipeline
    try:
        finetuned_selection_model = 'lukasellinger/evidence_selection_model-v4'
        finetuned_verification_model = 'lukasellinger/claim_verification_model-v4'
        translator = OpusMTTranslator()
        colon_sentence_connector = ColonSentenceConnector()
        phi_sentence_connector = PhiSentenceConnector()
        evid_fetcher = WikipediaEvidenceFetcher(offline=False)
        evid_selector = ModelEvidenceSelector(model_name=finetuned_selection_model)
        stm_verifier = ModelStatementVerifier(model_name=finetuned_verification_model)
        pipeline = Pipeline(translator=translator,
                            sent_connector=colon_sentence_connector,
                            claim_splitter=None,
                            evid_fetcher=evid_fetcher,
                            evid_selector=evid_selector,
                            stm_verifier=stm_verifier,
                            lang='de')
        return jsonify({'status': 'Pipeline loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/examples', methods=['GET'])
def get_examples():
    dataset_name = request.args.get('dataset')
    num_samples = int(request.args.get('num_samples', 10))  # Number of samples requested
    dataset_info = datasets.get(dataset_name, {})
    dataset = dataset_info.get('dataset', [])

    # Fetch 10 random examples
    examples = dataset.shuffle(seed=42).select(range(num_samples))
    formatted_examples = [
        {'word': example.get('word', ''), 'definition': example.get('claim', ''),
         'label': example.get('label', '')}
        for example in examples
    ]

    return jsonify(formatted_examples)


@app.route('/api/calculate_factuality', methods=['POST'])
def calculate_factuality():
    data = request.get_json()
    word = data.get('word')
    definition = data.get('definition')
    factuality = pipeline.verify(word, definition)['predicted']
    return jsonify({'factuality': factuality})


if __name__ == '__main__':
    app.run(debug=True)
