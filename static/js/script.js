let cursor = null;
let loading = false;
let finished = false;

/* ----- TAB NAVIGATION ----- */
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
    await viewCommentsPaging(id);
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

    let counter = 1;
    models.forEach((m) => {
      tbody.innerHTML += `
                <tr>
                    <td>${counter++}</td>
                    <td>${m.sy}</td>
                    <td>${m.semester}</td>
                    <td>${m.model_name}</td>
                    <td>${m.classifier}</td>
                    <td>${m.accuracy}%</td>
                    <td>${m.no_of_data}</td>
                    <td>${m.date_trained}</td>
                    <td>${m.remarks}</td>
                    <td>
                      <div>
                        <button class="btn btn-delete" onclick="deleteModel('${
                          m.id
                        }', this)">
                          Delete
                        </button>
                      </div>
                    </td>
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

    let counter = 1;
    result.forEach((m) => {
      tbody.innerHTML += `
        <tr>
          <td>${counter++}</td>
          <td class="multiline">
            <div class="tooltip" data-fulltext="${m.filename}">
              ${m.filename}
            </div>
          </td>
          <td>${m.no_of_data}</td>
          <td>
            <div class="tooltip" data-fulltext="${m.date_uploaded}">
              ${m.date_uploaded}
            </div>
          </td>
          <td class="multiline">
            <div class="tooltip" data-fulltext="${m.remarks ?? ""}">
              ${m.remarks ?? ""}
            </div>
          </td>
          <td>
          <div>
            <button id='${
              m.file_id
            }' class="btn btn-view" onclick="openCommentsTab('${m.file_id}')">
            Comments
            </button>
            <br/>
            </button>            
            <button class="btn btn-delete" onclick="deleteFile('${
              m.file_id
            }', this)">
            Delete
            </button>
          </div>
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

async function viewCommentsPaging(file_id = null) {
  const tbody = document.getElementById("comments-tbody");

  cursor = null;
  finished = false;
  let counter = 1;
  tbody.innerHTML = ""; // clear previous comments

  // Function to load comments
  async function loadComments(file_id) {
    if (loading || finished) return;

    loading = true;

    let url = `/comments-paging?limit=20`;
    if (file_id) {
      url += `&file_id=${file_id}`;
    }

    if (cursor) {
      url += `&cursor=${cursor}`;
    }

    try {
      const response = await fetch(url);
      const data = await response.json();

      if (data.comments.length === 0) {
        finished = true; // no more comments
        return;
      }

      data.comments.forEach((m) => {
        tbody.innerHTML += `
          <tr>
            <td>${counter++}</td>
            <td>${m.comment}</td>
            <td>${m.label}</td>
            <td>${m.remarks ?? ""}</td>
          </tr>
        `;
      });

      // Update cursor to the last comment's ID
      cursor = data.comments[data.comments.length - 1].id;
    } catch (err) {
      console.error("Failed to load comments:", err);
    } finally {
      loading = false;
    }
  }

  // Scroll listener for infinite scrolling
  window.addEventListener("scroll", () => {
    if (
      window.innerHeight + window.scrollY >=
      document.body.offsetHeight - 100
    ) {
      loadComments(file_id);
    }
  });

  // Initial load
  loadComments(file_id);
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

  const selectElement = document.getElementById("model-select");

  // Clear existing options
  selectElement.innerHTML = "";

  // Append fresh options
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_name;
    option.textContent = `${model.model_name} (${model.classifier} - ${model.accuracy}% accuracy)`;
    selectElement.appendChild(option);
  });
}

async function get_sentiments() {
  const modelSelect = document.getElementById("model-select");
  const commentsBox = document.getElementById("comments-box");
  const sentimentDisplay = document.getElementById("sentiment-result");
  const spinner = document.getElementById("sentiment-spinner");
  const sentimentBtn = document.getElementById("sentiment-btn");

  const model_name = modelSelect.value.trim();
  const comments = commentsBox.value.trim();

  if (!model_name) {
    alert("Please choose a model name.");
    return;
  }

  if (!comments) {
    alert("Please write a comment first.");
    return;
  }

  // Disable button while processing
  sentimentBtn.disabled = true;

  // Prepare display
  sentimentDisplay.style.maxHeight = "300px";
  sentimentDisplay.style.overflowY = "auto";
  sentimentDisplay.innerHTML = "";
  spinner.style.display = "block";

  const lines = comments
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  const payload = { model_name, lines: lines };

  try {
    const res = await fetch(`/predict-sentiment?model_name=${model_name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errorText = await res.text();
      sentimentDisplay.innerHTML = `Error: ${errorText}`;
      throw new Error("Failed to fetch /predict-sentiment.");
    }

    const data = await res.json();

    let counter = 1;
    for (const [comment, item] of Object.entries(data.sentiments || data)) {
      const color =
        item.sentiment === "positive"
          ? "blue"
          : item.sentiment === "negative"
          ? "red"
          : "black";

      sentimentDisplay.innerHTML += `
        <div style="font-size: 15px; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #ddd;">
          <span>${counter}. </span> ${comment} → <span style="color: ${color}; font-weight: bold;">${
        item.sentiment
      }</span>
          <br/>
          <span style="font-size: 14px; color: #0a21eeff;">Top Category:</span> ${
            item.top_category
          } <span style="margin: 2px; font-size: 12px;">
          <br/>
          <span style="font-size: 14px; color: #0f4404ff;">Categories:</span>
          <br/>
          <div style="margin-left: 15px; font-size: 12px;">
            ${Object.entries(item.category)
              .map(([cat, score]) => `${cat}: ${score}%`)
              .join(", ")}
          </div>
        </div>
      `;
      counter++;
    }
  } catch (error) {
    console.error(error);
  } finally {
    // Re-enable button and hide spinner
    sentimentBtn.disabled = false;
    spinner.style.display = "none";
  }
}

/* ----- COMMENT SEARCH ----- */
document
  .getElementById("comment-search")
  .addEventListener("input", function () {
    const filter = this.value.toLowerCase();
    const rows = document.querySelectorAll("#comments-tbody tr");

    rows.forEach((row) => {
      const comment = row.children[1].textContent.toLowerCase();
      row.style.display = comment.includes(filter) ? "" : "none";
    });
  });

async function deleteModel(model_id, button) {
  if (!confirm("Are you sure you want to delete this model?")) return;

  // Save original button content
  const originalContent = button.innerHTML;

  // Show spinner inside the button
  button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Deleting...`;
  button.disabled = true;

  try {
    const res = await fetch(`/delete-model/${model_id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete model.");

    // Remove the row from the table
    const row = button.closest("tr");
    if (row) row.remove();

    // Optionally, refresh the table instead:
    // await viewTrainedModels();
  } catch (err) {
    console.error(err);
    alert("Error deleting model.");

    // Restore original button
    button.innerHTML = originalContent;
    button.disabled = false;
  }
}

async function deleteFile(file_id, button) {
  if (!confirm("Are you sure you want to delete this file?")) return;

  const commentBtn = document.getElementById(file_id);
  commentBtn.disabled = true;

  // Save original button content
  const originalContent = button.innerHTML;

  // Show inline spinner
  button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Deleting...`;
  button.disabled = true;

  try {
    const res = await fetch(`/delete-file/${file_id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete file.");

    alert("File deleted successfully.");

    // Remove the row from the table (optional: faster than refreshing the whole table)
    const row = button.closest("tr");
    if (row) row.remove();

    // Or, if you prefer, refresh the list:
    // await viewUploadedFiles();
  } catch (err) {
    console.error(err);
    alert("Error deleting file.");

    // Restore original button if deletion fails
    button.innerHTML = originalContent;
    button.disabled = false;
    commentBtn.disabled = false;
  }
}

//Initialize by opening the Models tab
openTab("models");
