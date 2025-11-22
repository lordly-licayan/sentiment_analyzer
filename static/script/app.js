/* ----- TAB SWITCHING ----- */
function openTab(tabId) {
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((c) => c.classList.remove("active"));

  event.target.classList.add("active");
  document.getElementById(tabId).classList.add("active");
}

/* ----- CSV UPLOAD HANDLING ----- */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileName");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("dragover")
);

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");

  if (e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    if (!file.name.endsWith(".csv")) {
      alert("Only CSV files allowed!");
      return;
    }
    fileInput.files = e.dataTransfer.files;
    fileNameDisplay.textContent = "Selected: " + file.name;
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    if (!file.name.endsWith(".csv")) {
      alert("Only CSV files allowed!");
      fileInput.value = "";
      return;
    }
    fileNameDisplay.textContent = "Selected: " + file.name;
  }
});

async function uploadCSV() {
  const fileInput = document.getElementById("fileInput");
  const schoolYearSelect = document.getElementById("schoolYearSelect");
  const semesterSelect = document.getElementById("semesterSelect");
  const datasetTypeSelect = document.getElementById("datasetTypeSelect");

  // --- VALIDATIONS ---
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select a CSV file to upload.");
    return;
  }

  const file = fileInput.files[0];

  if (!file.name.toLowerCase().endsWith(".csv")) {
    alert("Invalid file type — only CSV files are allowed.");
    return;
  }

  if (!schoolYearSelect.value) {
    alert("Please select a School Year.");
    return;
  }

  if (!semesterSelect.value) {
    alert("Please select a Semester.");
    return;
  }

  if (!datasetTypeSelect.value) {
    alert("Please choose a Dataset Type (training/testing).");
    return;
  }

  // --- UPLOAD ---
  const formData = new FormData();
  formData.append("file", file); // CSV File
  formData.append("sy", schoolYearSelect.value); // School Year
  formData.append("semester", semesterSelect.value); // Semester
  formData.append("datasetType", datasetTypeSelect.value); // Dataset Type

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      alert("Upload failed.");
      return;
    }

    const data = await res.json();
    alert("Upload complete!");
    console.log(data);
  } catch (err) {
    console.error(err);
    alert("Error uploading file.");
  }
}

/* ----- VIEW BUTTONS (PLACEHOLDER) ----- */
function viewComments(fileId) {
  openTab("comments");
  alert("Load comments via API for File ID: " + fileId);
}

function viewReport(fileId) {
  openTab("report");
  alert("Load training report via API for File ID: " + fileId);
}
