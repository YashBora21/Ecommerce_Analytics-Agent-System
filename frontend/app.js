const API = "http://127.0.0.1:8000";
const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const status = document.querySelector("#status");
let threadId = sessionStorage.getItem("ecommerce-thread");

function addMessage(role, text, usage) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  if (usage) {
    const meta = document.createElement("small");
    meta.textContent = `${usage.input_tokens} in · ${usage.output_tokens} out · ${usage.total_tokens} total tokens`;
    article.appendChild(meta);
  }
  messages.appendChild(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API}/api/health`);
    if (!response.ok) throw new Error();
    status.textContent = "API connected";
    status.classList.add("online");
  } catch {
    status.textContent = "API unavailable";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage("user", question);
  input.value = "";
  send.disabled = true;
  send.textContent = "Working…";
  const pending = addMessage("assistant", "Analyzing the database…");

  try {
    const response = await fetch(`${API}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");

    threadId = data.thread_id;
    sessionStorage.setItem("ecommerce-thread", threadId);
    pending.remove();
    const answer = addMessage("assistant", data.answer, data.usage);
    if (data.chart) {
      const chart = document.createElement("div");
      chart.className = "chart";
      answer.appendChild(chart);
      if (window.Plotly) {
        Plotly.newPlot(chart, data.chart.data, data.chart.layout, { responsive: true });
      } else {
        chart.textContent = "Plotly could not load, so the chart is unavailable.";
      }
    }
  } catch (error) {
    pending.querySelector(".bubble").textContent = error.message;
    pending.classList.add("error");
  } finally {
    send.disabled = false;
    send.textContent = "Ask";
    input.focus();
  }
});

document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    input.focus();
  });
});

checkHealth();
