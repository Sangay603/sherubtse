document.addEventListener('DOMContentLoaded', function() {
    fetch('notebook.html')
        .then(response => response.text())
        .then(html => {
            document.getElementById('notebook-content').innerHTML = html;
        })
        .catch(error => {
            document.getElementById('notebook-content').innerHTML = '<p>Notebook output could not be loaded.</p>';
        });
}); 