document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-upload");
    const predictButton = document.getElementById("predict-button");
    const predictionsContainer = document.getElementById("predictions");
    const predictionsList = document.getElementById("predictions-list");

    uploadForm.addEventListener("submit", function (e) {
        e.preventDefault();

        // Get the selected files
        const files = fileInput.files;

        // You can perform file upload logic here if needed

        // For demonstration purposes, we'll display the selected file names
        predictionsList.innerHTML = "";
        for (let i = 0; i < files.length; i++) {
            const listItem = document.createElement("li");
            listItem.textContent = `File ${i + 1}: ${files[i].name}`;
            predictionsList.appendChild(listItem);
        }

        // Display the predictions container
        predictionsContainer.classList.remove("hidden");
    });
});
