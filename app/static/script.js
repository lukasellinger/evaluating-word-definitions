document.addEventListener('DOMContentLoaded', function() {
    const datasetSelect = document.getElementById('dataset');
    const examplesTable = document.getElementById('examplesTable');
    const wordInput = document.getElementById('word');
    const definitionTextarea = document.getElementById('definition');
    const calculateButton = document.getElementById('calculateButton');
    const resultParagraph = document.getElementById('result');
    const spinner = document.getElementById('spinner');
    const loadPipelineButton = document.getElementById('loadPipelineButton');
    const pipelineSpinner = document.getElementById('pipeline-spinner');
    let isPipelineLoaded = false;

    // Function to fetch and display examples
    function fetchExamples(dataset) {
        fetch(`/api/examples?dataset=${dataset}`)
            .then(response => response.json())
            .then(examples => {
                // Clear previous table rows
                examplesTable.innerHTML = '';

                // Populate the table with data
                examples.forEach(example => {
                    const row = document.createElement('tr');
                    row.classList.add('table-row-clickable'); // Add class for hover effect

                    const wordCell = document.createElement('td');
                    wordCell.textContent = example.word || 'N/A';
                    row.appendChild(wordCell);

                    const definitionCell = document.createElement('td');
                    definitionCell.textContent = example.definition || 'N/A';
                    row.appendChild(definitionCell);

                    const labelCell = document.createElement('td');
                    labelCell.textContent = example.label || 'N/A';
                    row.appendChild(labelCell);

                    // Add event listener to populate inputs on row click
                    row.addEventListener('click', function() {
                        wordInput.value = example.word || '';
                        definitionTextarea.value = example.definition || '';
                        autoResize(); // Adjust textarea height on row click
                    });

                    examplesTable.appendChild(row);
                });
            })
            .catch(error => console.error('Error fetching examples:', error));
    }

    // Function to auto-resize the definition textarea
    function autoResize() {
        definitionTextarea.style.height = 'auto';  // Reset the height
        definitionTextarea.style.height = definitionTextarea.scrollHeight + 'px';  // Adjust to scroll height
    }

    // Attach the auto-resize function to the input event
    definitionTextarea.addEventListener('input', autoResize);

    // Fetch examples when the dataset selection changes
    datasetSelect.addEventListener('change', function() {
        const dataset = this.value;
        fetchExamples(dataset);
    });

    // Load the pipeline
    loadPipelineButton.addEventListener('click', function() {
        // Show spinner and disable the button
        pipelineSpinner.style.display = 'inline-block';
        loadPipelineButton.disabled = true;

        fetch('/api/load_pipeline', {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide spinner and enable the calculate button
            pipelineSpinner.style.display = 'none';
            calculateButton.disabled = false;
            isPipelineLoaded = true;
            alert('Pipeline loaded successfully!');
        })
        .catch(error => {
            console.error('Error loading pipeline:', error);
            pipelineSpinner.style.display = 'none';
            loadPipelineButton.disabled = false;
            alert('Error loading pipeline. Please try again.');
        });
    });

    // Calculate factuality and display result
    calculateButton.addEventListener('click', function() {
        if (!isPipelineLoaded) {
            alert('Please load the pipeline first.');
            return;
        }

        // Validate inputs
        const word = wordInput.value.trim();
        const definition = definitionTextarea.value.trim();

        if (!word || !definition) {
            alert('Both Word and Definition fields are required.');
            return;
        }

        // Show spinner and hide result
        spinner.classList.add('show');
        resultParagraph.textContent = '';

        fetch('/api/calculate_factuality', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                word: word,
                definition: definition,
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide spinner and display result
            spinner.classList.remove('show');
            resultParagraph.textContent = `${data.factuality}`;
        })
        .catch(error => {
            console.error('Error calculating factuality:', error);
            spinner.classList.remove('show');
            resultParagraph.textContent = 'Error calculating factuality. Please try again.';
        });
    });

    // Initial load of examples from the default dataset
    const defaultDataset = datasetSelect.value;
    fetchExamples(defaultDataset);

    // Initial call to auto-resize function to adjust textarea height on page load
    autoResize();
});
