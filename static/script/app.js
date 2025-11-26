/* ----- TAB SWITCHING ----- */
// async function openTab(tabId) {
//   document
//     .querySelectorAll(".tab")
//     .forEach((t) => t.classList.remove("active"));
//   document
//     .querySelectorAll(".tab-content")
//     .forEach((c) => c.classList.remove("active"));

//   event.target.classList.add("active");
//   document.getElementById(tabId).classList.add("active");

//   if (tabId === "models") {
//     await viewTrainedModels();
//   }

//   if (tabId === "uploaded_files") {
//     await viewUploadedFiles();
//   }

//   if (tabId === "comments") {
//     await viewComments();
//   }
// }

async function openTab(tabId, id = null) {
  document
    .querySelectorAll(".tab")
    .forEach((tab) => tab.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.style.display = "none";
    content.classList.remove("active");
  });

  // Activate tab header
  const header = document.querySelector(`.tab[onclick="openTab('${tabId}')"]`);
  if (header) header.classList.add("active");

  // Show content
  const content = document.getElementById(tabId);
  if (content) {
    content.style.display = "block";
    content.classList.add("active");
  }

  if (tabId === "models") {
    await viewTrainedModels();
  }

  if (tabId === "uploaded_files") {
    await viewUploadedFiles();
  }

  if (tabId === "comments") {
    await viewComments(id);
  }

  if (tabId === "playground") {
    await viewPlayground();
  }
}

/* ----- DISPLAYING TRAINED MODELS ----- */
async function viewTrainedModels() {
  const tbody = document.getElementById("models-tbody");
  tbody.innerHTML = `<tr><td colspan="8">Loading...</td></tr>`;

  try {
    const res = await fetch("/trained-models");
    if (!res.ok) throw new Error("Failed to fetch");

    const models = await res.json();

    if (models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8">No trained models found.</td></tr>`;
      return;
    }

    tbody.innerHTML = ""; // Clear old rows

    models.forEach((m) => {
      tbody.innerHTML += `
                <tr>
                    <td>${m.sy}</td>
                    <td>${m.semester}</td>
                    <td>${m.model_name}</td>
                    <td>${m.classifier}</td>
                    <td>${m.accuracy}%</td>
                    <td>${m.no_of_data}</td>
                    <td>${m.date_trained}</td>
                    <td>${m.remarks}</td>
                </tr>
            `;
    });
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="8">Error loading models.</td></tr>`;
  }
}

/* ----- CSV UPLOAD HANDLING ----- */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileName");

document.getElementById("report-info").style.display = "none";

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
        document.getElementById("elapsedTime").innerText = data.elapsedTime;
        document.getElementById("report").innerText = JSON.stringify(
          data.report
        );
        document.getElementById("feedback").innerText = data.feedback;
        startTraining(false);
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

function isValidModelName(filename) {
  if (!filename) return false; // empty filename

  // Check for spaces
  if (filename.includes(" ")) {
    return false;
  }

  return true;
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

/* ----- UPLOADED FILES ----- */
async function viewUploadedFiles() {
  const tbody = document.getElementById("uploaded-files-tbody");
  tbody.innerHTML = `<tr><td colspan="5">Loading...</td></tr>`;

  try {
    const res = await fetch("/uploaded-files");
    if (!res.ok) throw new Error("Failed to fetch /uploaded-files");

    const result = await res.json();

    if (result.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5">No uploaded files found.</td></tr>`;
      return;
    }

    tbody.innerHTML = ""; // Clear table

    result.forEach((m) => {
      tbody.innerHTML += `
        <tr>
          <td>${m.filename}</td>
          <td>${m.no_of_data}</td>
          <td>${m.date_uploaded}</td>
          <td>${m.remarks ?? ""}</td>
          <td>
            <button class="btn btn-view" onclick="openCommentsTab('${
              m.file_id
            }')">
            Comments
          </button>
          </td>
        </tr>
      `;
    });
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="5">Error loading uploaded files.</td></tr>`;
  }
}

/* ----- VIEW COMMENTS ----- */
async function openCommentsTab(file_id) {
  openTab("comments", file_id);
}

async function viewComments(file_id = null) {
  const tbody = document.getElementById("comments-tbody");
  tbody.innerHTML = `<tr><td colspan="5">Loading...</td></tr>`;

  try {
    let res;

    if (file_id) {
      res = await fetch(`/comments?file_id=${file_id}`);
    } else {
      res = await fetch("/comments");
    }

    if (!res.ok) throw new Error("Failed to fetch /comments");

    const result = await res.json();

    if (result.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5">No comments found.</td></tr>`;
      return;
    }

    tbody.innerHTML = ""; // Clear table

    result.forEach((m) => {
      tbody.innerHTML += `
        <tr>
          <td>${m.comment}</td>
          <td>${m.label}</td>
          <td>${m.remarks ?? ""}</td>
        </tr>
      `;
    });
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="5">Error loading comments.</td></tr>`;
  }
}

/* ----- VIEW PLAYGROUND ----- */
async function viewPlayground() {
  const res = await fetch("/trained-models");

  if (!res.ok) throw new Error("Failed to fetch /trained-models");

  const models = await res.json();
  if (models.length === 0) {
    alert("No trained models available!");
    return;
  }

  // Get the select element
  const selectElement = document.getElementById("model-select");

  // Populate only with model_name
  models
    .map((model) => {
      const option = document.createElement("option");
      option.value = model.model_name;
      option.textContent = `${model.model_name} (${model.classifier} - ${model.accuracy}% accuracy)`;
      return option;
    })
    .forEach((option) => selectElement.appendChild(option));
}

async function get_sentiments() {
  const model_name = document.getElementById("model-select").value.trim();
  const comments = document.getElementById("comments-box").value.trim();

  if (!model_name) {
    alert("Please choose a model name.");
  } else if (!comments) {
    alert("Please write a comment first.");
  }
  const sentiment_display = document.getElementById("sentiment-result");
  const spinner = document.getElementById("sentiment-spinner");

  sentiment_display.innerHTML = "";
  spinner.style.display = "block";

  const res = await fetch(`/predict-sentiment?model_name=${model_name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: comments }),
  });

  if (!res.ok) throw new Error("Failed to fetch /predict-sentiment.");

  const data = await res.json();

  spinner.style.display = "none";

  for (const [comment, sentiment] of Object.entries(data)) {
    const color =
      sentiment === "positive"
        ? "blue"
        : sentiment === "negative"
        ? "red"
        : "black";

    sentiment_display.innerHTML += `
      <div style="font-size: 15px; margin-bottom: 8px;">
        ${comment} → <span style="color: ${color};">${sentiment}</span>
      </div>
    `;
  }
}
