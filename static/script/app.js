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

function updateProgressUI(percent, label) {
  const bar = document.getElementById("progressBar");
  const text = document.getElementById("progressPercent");
  const labelText = document.getElementById("progressLabel");

  bar.style.width = percent;
  text.innerText = percent;
  labelText.innerText = label;
}

function pollTraining(job_id) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/training-status/${job_id}`);

      if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`);
      }

      const data = await res.json();

      document.getElementById("status").innerText = data.status;
      updateProgressUI(data.progress, data.message);

      if (data.status === "Complete") {
        clearInterval(interval);
        document.getElementById("progressLabel").innerText = "";

        document.getElementById("report-info").style.display = "block";
        document.getElementById("message").innerText = data.message;
        document.getElementById("accuracy").innerText = data.accuracy;
        document.getElementById("report").innerText = JSON.stringify(
          data.report
        );
        document.getElementById("feedback").innerText = data.feedback;
        startTraining(false);
        alert("Training completed!");
      }

      if (data.status?.startsWith("Error")) {
        clearInterval(interval);
        startTraining(false);
        updateProgressUI(0, data.message);
        console.error("Error:", data.message);
      }
    } catch (err) {
      // Handle HTTP 500, network drops, JSON parsing failures
      clearInterval(interval);
      startTraining(false);
      console.error("Polling error:", err);
    }
  }, 1000);
}

function startTraining(enable) {
  const btn = document.getElementById("trainBtn");
  btn.disabled = enable;

  if (enable) {
    btn.innerText = "Training...";
  } else {
    btn.innerText = "Train";
  }
}

async function train_model() {
  const modelName = document.getElementById("modelName").value.trim();

  if (!isValidModelName(modelName)) {
    alert("Invalid model name. Ensure it has no spaces.");
    return;
  }

  const fileInput = document.getElementById("fileInput");
  const schoolYearSelect = document.getElementById("schoolYearSelect");
  const semesterSelect = document.getElementById("semesterSelect");

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

  // --- TRAINING MODEL ---
  const formData = new FormData();
  formData.append("modelName", modelName); // Model Name
  formData.append("file", file); // CSV File
  formData.append("sy", schoolYearSelect.value); // School Year
  formData.append("semester", semesterSelect.value); // Semester
  formData.append("classifierModel", classifierModel.value); // classifier Model

  startTraining(true);
  document.getElementById("report-info").style.display = "none";

  // Show progress UI
  document.getElementById("progressContainer").style.display = "block";
  updateProgressUI(0, "Starting...");

  document.getElementById("accuracy").innerText = "";
  document.getElementById("report").innerText = "";

  try {
    const res = await fetch("/train_model", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      alert("Training failed.");
      return;
    }

    const data = await res.json();
    const job_id = data.job_id;
    pollTraining(job_id);

    console.log(data);
  } catch (err) {
    console.error(err);
    startTraining(false);
    alert("Error training a model.");
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

function isValidModelName(filename) {
  if (!filename) return false; // empty filename

  // Check for spaces
  if (filename.includes(" ")) {
    return false;
  }

  // Check if it ends with .pkl (case-insensitive)
  // if (!filename.toLowerCase().endsWith(".pkl")) {
  //   return false;
  // }

  return true;
}

document.getElementById("report-info").style.display = "none";
