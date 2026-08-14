const state = {
  sourceDir: localStorage.getItem("scenelog.sourceDir") || "",
  project: null,
  taskStatus: "idle",
  pollTimer: null,
  setup: null,
};

const elements = {
  setupBanner: document.querySelector("#setup-banner"),
  setupState: document.querySelector("#setup-state"),
  setupTitle: document.querySelector("#setup-title"),
  setupDescription: document.querySelector("#setup-description"),
  setupToggle: document.querySelector("#setup-toggle"),
  setupDetails: document.querySelector("#setup-details"),
  setupChecks: document.querySelector("#setup-checks"),
  sourceDir: document.querySelector("#source-dir"),
  chooseFolder: document.querySelector("#choose-folder"),
  loadProject: document.querySelector("#load-project"),
  projectSummary: document.querySelector("#project-summary"),
  peopleCount: document.querySelector("#people-count"),
  personForm: document.querySelector("#person-form"),
  personName: document.querySelector("#person-name"),
  personPhotos: document.querySelector("#person-photos"),
  uploadCaption: document.querySelector("#upload-caption"),
  photoPreviews: document.querySelector("#photo-previews"),
  addPerson: document.querySelector("#add-person"),
  peopleList: document.querySelector("#people-list"),
  runCaption: document.querySelector("#run-caption"),
  taskBadge: document.querySelector("#task-badge"),
  mediaCount: document.querySelector("#media-count"),
  processedCount: document.querySelector("#processed-count"),
  failedCount: document.querySelector("#failed-count"),
  visionEnabled: document.querySelector("#vision-enabled"),
  peopleEnabled: document.querySelector("#people-enabled"),
  startProcess: document.querySelector("#start-process"),
  stopProcess: document.querySelector("#stop-process"),
  downloadExcel: document.querySelector("#download-excel"),
  clearLog: document.querySelector("#clear-log"),
  processLog: document.querySelector("#process-log"),
  searchForm: document.querySelector("#search-form"),
  searchInput: document.querySelector("#search-input"),
  searchCount: document.querySelector("#search-count"),
  searchResults: document.querySelector("#search-results"),
  toast: document.querySelector("#toast"),
};

elements.sourceDir.value = state.sourceDir;
const buttonContents = new WeakMap();

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) {
    throw new Error(payload?.error || `请求失败 (${response.status})`);
  }
  return payload;
}

function showToast(message, type = "info") {
  elements.toast.textContent = message;
  elements.toast.className = `toast visible ${type === "error" ? "error" : ""}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.className = "toast";
  }, 3600);
}

function setBusy(button, busy, label) {
  if (!buttonContents.has(button)) {
    buttonContents.set(button, button.innerHTML);
  }
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
  if (busy) {
    button.textContent = label;
  } else {
    button.innerHTML = buttonContents.get(button);
  }
}

function currentSourceDir() {
  return elements.sourceDir.value.trim();
}

function renderSetup(status) {
  state.setup = status;
  elements.setupBanner.classList.toggle("ready", status.ready);
  elements.setupBanner.classList.toggle("needs-setup", !status.ready);
  elements.setupState.className = `setup-state ${status.ready ? "ready" : "warning"}`;
  elements.setupState.textContent = status.ready ? "可以开始" : "需要配置";
  elements.setupTitle.textContent = status.ready
    ? "本机环境已经准备好"
    : `已完成 ${status.completed}/${status.total} 项环境配置`;
  elements.setupDescription.textContent = status.ready
    ? `Scenelog v${status.version} · ${status.platform.machine} · 可用空间 ${status.free_disk_gb} GB`
    : "展开详情查看缺失组件；未安装的可选能力不会阻止基础功能启动。";
  elements.setupChecks.replaceChildren();
  for (const check of status.checks) {
    const row = document.createElement("div");
    row.className = `setup-check ${check.ready ? "ready" : "missing"}`;
    const icon = document.createElement("span");
    icon.className = "setup-check-icon";
    icon.textContent = check.ready ? "✓" : "!";
    const copy = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = check.label;
    if (!check.required) {
      const optional = document.createElement("small");
      optional.className = "optional-label";
      optional.textContent = "可选";
      label.append(optional);
    }
    const detail = document.createElement("p");
    detail.textContent = check.detail;
    copy.append(label, detail);
    row.append(icon, copy);
    elements.setupChecks.append(row);
  }
}

async function loadSetupStatus() {
  try {
    renderSetup(await request("/api/setup/status"));
  } catch (error) {
    elements.setupState.className = "setup-state warning";
    elements.setupState.textContent = "检查失败";
    elements.setupTitle.textContent = "无法读取本机环境状态";
    elements.setupDescription.textContent = error.message;
  }
}

async function loadProject(showMessage = true) {
  const sourceDir = currentSourceDir();
  if (!sourceDir) {
    showToast("请先选择素材目录", "error");
    return;
  }
  setBusy(elements.loadProject, true, "载入中");
  try {
    const query = new URLSearchParams({ source_dir: sourceDir });
    state.project = await request(`/api/project?${query}`);
    state.sourceDir = state.project.source_dir;
    elements.sourceDir.value = state.sourceDir;
    localStorage.setItem("scenelog.sourceDir", state.sourceDir);
    renderProject(state.project);
    if (showMessage) {
      showToast(`已载入 ${state.project.media_count} 条素材`);
    }
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setBusy(elements.loadProject, false, "");
  }
}

function renderProject(project) {
  elements.projectSummary.textContent =
    `${project.media_count} 条素材 · 输出到 ${project.output_dir}`;
  elements.mediaCount.textContent = project.media_count;
  elements.processedCount.textContent = project.processed_count;
  elements.failedCount.textContent = project.failed_count;
  elements.runCaption.textContent = project.media_count
    ? `已发现 ${project.media_count} 条素材，将分析人物动作角色`
    : "目录中没有支持的视频或音频";
  elements.startProcess.disabled =
    !project.media_count || ["running", "stopping"].includes(state.taskStatus);
  elements.peopleCount.textContent = project.people.length;
  renderPeople(project.people);

  if (project.excel_ready) {
    const query = new URLSearchParams({ source_dir: project.source_dir });
    elements.downloadExcel.href = `/api/download/excel?${query}`;
    elements.downloadExcel.setAttribute("aria-disabled", "false");
  } else {
    elements.downloadExcel.href = "#";
    elements.downloadExcel.setAttribute("aria-disabled", "true");
  }
}

function renderPeople(people) {
  elements.peopleList.replaceChildren();
  if (!people.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "尚未登记关键人物";
    elements.peopleList.append(empty);
    return;
  }

  for (const person of people) {
    const row = document.createElement("div");
    row.className = "person-row";

    const avatar = person.thumbnail_url
      ? document.createElement("img")
      : document.createElement("span");
    if (person.thumbnail_url) {
      avatar.src = person.thumbnail_url;
      avatar.alt = `${person.name}头像`;
    } else {
      avatar.className = "person-avatar";
      avatar.textContent = person.name.slice(0, 1);
    }

    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = person.name;
    const detail = document.createElement("small");
    const voiceText = person.voice_duration
      ? `${Math.round(person.voice_duration)} 秒声纹`
      : "声纹未登记";
    detail.textContent =
      `${person.reference_count} 张照片 · ${voiceText} · ${person.sample_count} 次命中`;
    info.append(name, detail);

    const voiceLabel = document.createElement("label");
    voiceLabel.className =
      `voice-person ${person.voice_duration ? "registered" : ""}`;
    voiceLabel.title = `为 ${person.name} 添加声音样本`;
    voiceLabel.setAttribute("aria-label", `为 ${person.name} 添加声音样本`);
    voiceLabel.textContent = person.voice_duration ? "补充声纹" : "添加声纹";
    const voiceInput = document.createElement("input");
    voiceInput.type = "file";
    voiceInput.accept = "audio/*,video/mp4,video/quicktime";
    voiceInput.multiple = true;
    voiceInput.addEventListener("change", () => addPersonVoice(person, voiceInput));
    voiceLabel.append(voiceInput);

    const remove = document.createElement("button");
    remove.className = "delete-person";
    remove.type = "button";
    remove.title = `删除 ${person.name}`;
    remove.setAttribute("aria-label", `删除 ${person.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => deletePerson(person));

    const actions = document.createElement("div");
    actions.className = "person-actions";
    actions.append(voiceLabel, remove);
    row.append(avatar, info, actions);
    elements.peopleList.append(row);
  }
}

async function addPersonVoice(person, input) {
  if (!state.project || !input.files.length) {
    return;
  }
  const files = [...input.files];
  if (files.length > 5) {
    showToast("每次最多上传 5 段声音样本", "error");
    input.value = "";
    return;
  }
  const label = input.closest(".voice-person");
  label.classList.add("busy");
  try {
    const samples = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        data: await fileToDataURL(file),
      })),
    );
    const result = await request("/api/people/voice", {
      method: "POST",
      body: JSON.stringify({
        source_dir: state.sourceDir,
        person_id: person.id,
        samples,
      }),
    });
    state.project.people = result.people;
    renderProject(state.project);
    showToast(`${result.message}，下次处理将自动识别说话人`);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    label.classList.remove("busy");
    input.value = "";
  }
}

async function deletePerson(person) {
  if (!window.confirm(`确定删除“${person.name}”及其参考照片吗？`)) {
    return;
  }
  try {
    const result = await request("/api/people/delete", {
      method: "POST",
      body: JSON.stringify({
        source_dir: state.sourceDir,
        person_id: person.id,
      }),
    });
    state.project.people = result.people;
    renderProject(state.project);
    showToast(`已删除 ${person.name}`);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderPhotoPreviews() {
  elements.photoPreviews.replaceChildren();
  const files = [...elements.personPhotos.files];
  elements.uploadCaption.textContent = files.length
    ? `已选择 ${files.length} 张照片`
    : "支持 JPG、PNG、WebP";
  for (const file of files.slice(0, 5)) {
    const image = document.createElement("img");
    image.alt = file.name;
    image.src = URL.createObjectURL(file);
    image.addEventListener("load", () => URL.revokeObjectURL(image.src), {
      once: true,
    });
    elements.photoPreviews.append(image);
  }
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`无法读取照片: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function addPerson(event) {
  event.preventDefault();
  if (!state.project) {
    showToast("请先载入素材目录", "error");
    return;
  }
  const files = [...elements.personPhotos.files];
  if (!files.length) {
    showToast("请至少选择一张人物照片", "error");
    return;
  }
  setBusy(elements.addPerson, true, "正在识别人脸");
  try {
    const photos = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        data: await fileToDataURL(file),
      })),
    );
    const result = await request("/api/people", {
      method: "POST",
      body: JSON.stringify({
        source_dir: state.sourceDir,
        name: elements.personName.value,
        photos,
      }),
    });
    state.project.people = result.people;
    renderProject(state.project);
    elements.personForm.reset();
    renderPhotoPreviews();
    showToast(result.message);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setBusy(elements.addPerson, false, "");
  }
}

async function startProcess() {
  if (!state.project) {
    showToast("请先载入素材目录", "error");
    return;
  }
  try {
    const task = await request("/api/process", {
      method: "POST",
      body: JSON.stringify({
        source_dir: state.sourceDir,
        options: {
          vision: elements.visionEnabled.checked,
          people: elements.peopleEnabled.checked,
        },
      }),
    });
    renderTask(task);
    startPolling();
    showToast("处理任务已启动");
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function stopProcess() {
  try {
    const task = await request("/api/process/stop", {
      method: "POST",
      body: "{}",
    });
    renderTask(task);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function taskLabel(status) {
  return {
    idle: "未运行",
    running: "运行中",
    stopping: "停止中",
    stopped: "已停止",
    completed: "已完成",
    failed: "失败",
  }[status] || status;
}

function renderTask(task) {
  const wasRunning = ["running", "stopping"].includes(state.taskStatus);
  state.taskStatus = task.status;
  const running = ["running", "stopping"].includes(task.status);
  document.body.classList.toggle("task-running", running);
  elements.taskBadge.textContent = taskLabel(task.status);
  elements.taskBadge.className =
    `task-badge ${task.status === "stopping" ? "running" : task.status}`;
  elements.startProcess.disabled = running || !state.project?.media_count;
  elements.stopProcess.disabled = !running;
  if (task.logs?.length) {
    const atBottom =
      elements.processLog.scrollHeight -
        elements.processLog.scrollTop -
        elements.processLog.clientHeight <
      50;
    elements.processLog.textContent = task.logs.join("\n");
    if (atBottom || running) {
      elements.processLog.scrollTop = elements.processLog.scrollHeight;
    }
  }
  if (wasRunning && !running) {
    loadProject(false);
    showToast(
      task.status === "completed" ? "素材处理完成" : `任务${taskLabel(task.status)}`,
      task.status === "failed" ? "error" : "info",
    );
  }
}

async function pollTask() {
  try {
    const task = await request("/api/task");
    renderTask(task);
    if (!["running", "stopping"].includes(task.status)) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch (error) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
    showToast(error.message, "error");
  }
}

function startPolling() {
  if (!state.pollTimer) {
    state.pollTimer = window.setInterval(pollTask, 1200);
  }
}

async function search(event) {
  event.preventDefault();
  if (!state.project) {
    showToast("请先载入素材目录", "error");
    return;
  }
  const queryText = elements.searchInput.value.trim();
  if (!queryText) {
    return;
  }
  try {
    const query = new URLSearchParams({
      source_dir: state.sourceDir,
      q: queryText,
    });
    const payload = await request(`/api/search?${query}`);
    renderSearchResults(payload.results);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderSearchResults(results) {
  elements.searchResults.replaceChildren();
  elements.searchCount.textContent = `${results.length} 条结果`;
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "没有找到匹配结果";
    elements.searchResults.append(empty);
    return;
  }

  const labels = {
    audio: "语音",
    vision: "画面",
    people: "人物",
    identity: "身份事件",
  };
  for (const result of results) {
    const row = document.createElement("article");
    row.className = "search-result";
    const time = document.createElement("time");
    time.textContent = `${result.start_time}–${result.end_time}`;
    const file = document.createElement("strong");
    file.textContent = result.rel_path;
    file.title = result.rel_path;
    const text = document.createElement("p");
    text.textContent = result.text;
    const source = document.createElement("span");
    source.className = "result-source";
    source.textContent = labels[result.source] || result.source;
    time.append(source);
    row.append(time, file, text);
    elements.searchResults.append(row);
  }
}

elements.chooseFolder.addEventListener("click", async () => {
  setBusy(elements.chooseFolder, true, "选择中");
  try {
    const result = await request("/api/folder-picker", {
      method: "POST",
      body: "{}",
    });
    if (!result.cancelled) {
      elements.sourceDir.value = result.source_dir;
      await loadProject();
    }
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setBusy(elements.chooseFolder, false, "");
  }
});

elements.loadProject.addEventListener("click", () => loadProject());
elements.sourceDir.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadProject();
  }
});
elements.personPhotos.addEventListener("change", renderPhotoPreviews);
elements.personForm.addEventListener("submit", addPerson);
elements.startProcess.addEventListener("click", startProcess);
elements.stopProcess.addEventListener("click", stopProcess);
elements.clearLog.addEventListener("click", () => {
  elements.processLog.textContent = "日志显示已清空，不影响后台任务。";
});
elements.searchForm.addEventListener("submit", search);
elements.setupToggle.addEventListener("click", () => {
  const expanded = elements.setupToggle.getAttribute("aria-expanded") === "true";
  elements.setupToggle.setAttribute("aria-expanded", String(!expanded));
  elements.setupToggle.textContent = expanded ? "查看详情" : "收起详情";
  elements.setupDetails.hidden = expanded;
});

loadSetupStatus();
pollTask();
if (state.sourceDir) {
  loadProject(false);
}
