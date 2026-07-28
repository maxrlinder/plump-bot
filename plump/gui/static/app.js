const setup = document.querySelector("#setup");
const table = document.querySelector("#table");
const form = document.querySelector("#setup-form");
const gameModeInput = document.querySelector("#game-mode");
const opponentsInput = document.querySelector("#opponents");
const handSizeInput = document.querySelector("#hand-size");
const minHandSizeInput = document.querySelector("#min-hand-size");
const maxHandSizeInput = document.querySelector("#max-hand-size");
const bidPositionInput = document.querySelector("#bid-position");
const newRoundButton = document.querySelector("#new-round");
const liveProbabilitiesInput = document.querySelector("#live-probabilities");

let state = null;
let botAdvanceTimer = null;
let advancingBot = false;
const CARD_PLAY_DELAY_MS = 1200;
const BID_DELAY_MS = 350;

function maxHandSize() {
  const players = Number(opponentsInput.value) + 1;
  return Math.min(10, Math.floor(52 / players));
}

function refreshSetupBounds() {
  const players = Number(opponentsInput.value) + 1;
  handSizeInput.max = String(maxHandSize());
  handSizeInput.min = "3";
  if (Number(handSizeInput.value) > maxHandSize()) {
    handSizeInput.value = String(maxHandSize());
  }
  bidPositionInput.innerHTML = "";
  for (let i = 1; i <= players; i += 1) {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${ordinal(i)} to bid`;
    bidPositionInput.append(option);
  }
  refreshFullGameBounds();
}

function refreshFullGameBounds() {
  const maximum = Math.min(10, Number(maxHandSizeInput.value));
  maxHandSizeInput.value = String(Math.max(5, maximum));
  minHandSizeInput.max = String(Number(maxHandSizeInput.value) - 1);
  if (Number(minHandSizeInput.value) >= Number(maxHandSizeInput.value)) {
    minHandSizeInput.value = String(Number(maxHandSizeInput.value) - 1);
  }
}

function refreshGameMode() {
  const fullGame = gameModeInput.value === "game";
  document.querySelector("#single-round-field").classList.toggle("hidden", fullGame);
  document.querySelector("#maximum-round-field").classList.toggle("hidden", !fullGame);
  document.querySelector("#minimum-round-field").classList.toggle("hidden", !fullGame);
  document.querySelector("#start-button").textContent = fullGame ? "Start Full Game" : "Start Round";
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  try {
    state = await postJson("/api/new", {
      mode: data.get("mode"),
      opponents: Number(data.get("opponents")),
      hand_size: Number(data.get("hand_size")),
      min_hand_size: Number(data.get("min_hand_size")),
      max_hand_size: Number(data.get("max_hand_size")),
      bid_position: Number(data.get("bid_position")),
      show_probabilities: data.has("show_probabilities"),
      seed: data.get("seed"),
    });
    setup.classList.add("hidden");
    table.classList.remove("hidden");
    render();
  } catch (error) {
    showSetupError(error.message);
  }
});

newRoundButton.addEventListener("click", () => {
  if (botAdvanceTimer) {
    window.clearTimeout(botAdvanceTimer);
    botAdvanceTimer = null;
  }
  state = null;
  table.classList.add("hidden");
  setup.classList.remove("hidden");
});

opponentsInput.addEventListener("change", refreshSetupBounds);
gameModeInput.addEventListener("change", refreshGameMode);
maxHandSizeInput.addEventListener("change", refreshFullGameBounds);
liveProbabilitiesInput.addEventListener("change", updateProbabilityVisibility);

function render() {
  if (!state) return;
  liveProbabilitiesInput.checked = Boolean(state.show_probabilities);
  liveProbabilitiesInput.disabled = !state.model_checkpoint;
  document.querySelector("#round-eyebrow").textContent =
    state.mode === "game" ? `Full Game · Round ${state.round_number}` : "Single Round";
  document.querySelector("#phase-title").textContent = phaseTitle(state);
  document.querySelector("#round-pill").textContent = `Round ${state.round_number} / ${state.total_rounds}`;
  document.querySelector("#trump-pill").textContent = `Trump: ${state.trump_label}`;
  document.querySelector("#hand-pill").textContent = `${state.hand_size} cards`;
  document.querySelector("#turn-pill").textContent = state.game_over
    ? "Game over"
    : state.round_over
      ? "Round complete"
    : state.human_turn
      ? "Your turn"
      : `${playerName(state.current_player)} thinking`;

  renderSchedule();
  renderPlayers();
  renderTrick();
  renderActions();
  renderHand();
  renderScoreboard();
  renderLog();
  maybeScheduleBotAdvance();
}

function renderSchedule() {
  const root = document.querySelector("#schedule");
  root.innerHTML = "";
  if (state.mode !== "game") {
    root.classList.add("hidden");
    return;
  }
  root.classList.remove("hidden");
  state.schedule.forEach((handSize, index) => {
    const roundNumber = index + 1;
    const item = document.createElement("span");
    const status = roundNumber < state.round_number
      ? " complete"
      : roundNumber === state.round_number
        ? " current"
        : "";
    item.className = `schedule-round${status}`;
    item.title = `Round ${roundNumber}: ${handSize} cards`;
    item.textContent = handSize;
    root.append(item);
  });
}

function renderPlayers() {
  const root = document.querySelector("#players");
  root.innerHTML = "";
  for (const player of state.players) {
    const item = document.createElement("article");
    item.className = `player${player.is_current ? " current" : ""}`;
    item.innerHTML = `
      <div class="player-top">
        <span class="player-name">${escapeHtml(player.name)}</span>
        <span class="badge">${player.cards} cards</span>
      </div>
      <div class="player-stats">
        <span>Bid: <strong>${player.bid ?? "—"}</strong></span>
        <span>Tricks: <strong>${player.tricks}</strong></span>
        <span>Score: <strong>${player.score}</strong></span>
      </div>
      ${player.is_human ? predictionHtml(player) : ""}
    `;
    root.append(item);
  }
}

function renderTrick() {
  const root = document.querySelector("#trick");
  root.innerHTML = "";
  root.style.setProperty("--player-count", String(state.players.length));
  const currentTrickIsEmpty = state.current_trick && !state.current_trick.plays.length;
  // The engine creates the next empty trick as soon as the prior one ends.
  // Keep the completed trick visible during the normal bot delay so its final
  // card and winner can actually be seen.
  const trick = currentTrickIsEmpty && state.last_trick
    ? state.last_trick
    : state.current_trick;
  const led = trick?.led_suit ? suitTitle(trick.led_suit) : "waiting for lead";
  document.querySelector("#leader-label").textContent = trick ? `Leader: ${playerName(trick.leader)} · Led: ${led}` : "";
  const players = trick ? trickOrder(trick.leader, state.players.length) : state.players.map((player) => player.id);
  for (const player of players) {
    const play = trick?.plays.find((row) => row.player === player);
    const won = trick?.winner === player;
    const slot = document.createElement("div");
    slot.className = `trick-slot${won ? " winner" : ""}`;
    slot.innerHTML = `
      <div class="slot-name">${play ? `${play.position + 1}. ` : ""}${playerName(player)}</div>
      ${play ? cardHtml(play.card) : '<div class="card empty">·</div>'}
      ${won ? '<div class="winner-label">Winner</div>' : ""}
    `;
    root.append(slot);
  }
}

function renderActions() {
  const root = document.querySelector("#action-panel");
  root.innerHTML = "";
  if (state.game_over) {
    const winners = state.winner_ids.map(playerName).join(" and ");
    const highScore = Math.max(...state.players.map((player) => player.score));
    root.innerHTML = `
      <h2>Game Complete</h2>
      <p class="result-callout">${escapeHtml(winners)} won with ${highScore} points.</p>
      ${roundScorecardHtml(state.round_scores)}
    `;
    return;
  }
  if (state.round_over) {
    root.innerHTML = `
      <h2>Round ${state.round_number} Complete</h2>
      ${roundScorecardHtml(state.round_scores)}
      <button id="next-round" class="primary inline-primary" type="button">Next Round</button>
    `;
    document.querySelector("#next-round").addEventListener("click", nextRound);
    return;
  }
  if (!state.human_turn) {
    root.innerHTML = `<h2>Waiting</h2><p class="subtle">${playerName(state.current_player)} is to act.</p>`;
    return;
  }
  if (state.phase === "bidding") {
    const modelHint = state.model_action_probabilities
      ? " Model policy probabilities from your POV are shown as decimals."
      : "";
    root.innerHTML = `<h2>Your Bid</h2><p class="subtle">Choose a legal bid. The forbidden last bid is omitted automatically.${modelHint}</p>`;
    const buttons = document.createElement("div");
    buttons.className = "bid-buttons";
    for (const bid of state.legal_bids) {
      const action = actionProbabilityForBid(bid);
      const button = document.createElement("button");
      button.className = `bid-button${action?.is_best ? " model-best" : ""}`;
      button.innerHTML = `
        <span>${bid}</span>
        ${action ? `<small>p=${formatProbability(action.probability)}</small>` : ""}
      `;
      button.addEventListener("click", () => submitBid(bid));
      buttons.append(button);
    }
    root.append(buttons);
    return;
  }
  if (state.phase === "playing") {
    const ledSuit = state.current_trick?.led_suit ? suitTitle(state.current_trick.led_suit) : null;
    const hint = ledSuit
      ? `Led suit is ${ledSuit}. If you have none, every card is legal.`
      : "You are leading and may play any card.";
    const modelHint = state.model_action_probabilities
      ? " Policy probabilities below use only your legal cards."
      : "";
    root.innerHTML = `<h2>Your Play</h2><p class="subtle">${hint}${modelHint}</p>`;
  }
}

function renderHand() {
  const root = document.querySelector("#hand");
  root.innerHTML = "";
  document.querySelector("#legal-hint").textContent =
    state.phase === "playing" && state.human_turn
      ? "Click a legal card · ★ is the model's top choice"
      : "Always visible";
  for (const card of state.my_hand) {
    const action = actionProbabilityForCard(card.key);
    const button = document.createElement("button");
    const legal = card.legal && state.human_turn && state.phase === "playing";
    button.className = `card-button${legal ? " legal" : ""}${action?.is_best ? " model-best" : ""}`;
    button.disabled = !(card.legal && state.human_turn && state.phase === "playing");
    button.innerHTML = `
      ${cardHtml(card)}
      ${action ? `<span class="policy-prob">${action.is_best ? "★ " : ""}p=${formatProbability(action.probability)}</span>` : ""}
    `;
    button.addEventListener("click", () => submitPlay(card));
    root.append(button);
  }
}

function renderScoreboard() {
  const scoreboard = document.querySelector("#scoreboard");
  const ranked = state.players.slice().sort((left, right) => right.score - left.score || left.id - right.id);
  scoreboard.innerHTML = ranked
    .map((player, index) => `
      <div class="score-row${state.winner_ids.includes(player.id) ? " winner" : ""}">
        <span>${index + 1}. ${escapeHtml(player.name)}</span>
        <strong>${player.score}</strong>
      </div>
    `)
    .join("");

  const results = document.querySelector("#round-results");
  if (!state.completed_rounds.length) {
    results.innerHTML = '<p class="empty-results">No completed rounds yet.</p>';
    return;
  }
  results.innerHTML = state.completed_rounds
    .slice()
    .reverse()
    .slice(0, 3)
    .map((round) => {
      const points = state.players
        .map((player) => `${playerName(player.id)} +${round.scores[player.id] ?? 0}`)
        .join(" · ");
      return `
        <div class="round-result">
          <strong>R${round.round_number} · ${round.hand_size} cards</strong>
          <span>${escapeHtml(points)}</span>
        </div>
      `;
    })
    .join("");
}

function renderLog() {
  const messages = document.querySelector("#messages");
  messages.innerHTML = "";
  for (const message of state.messages.slice().reverse().slice(0, 3)) {
    const row = document.createElement("div");
    row.className = "message";
    row.textContent = message;
    messages.append(row);
  }

  const events = document.querySelector("#events");
  events.innerHTML = "";
  for (const event of state.event_log.slice().reverse().slice(0, 3)) {
    const row = document.createElement("div");
    row.className = "event";
    row.textContent = eventText(event);
    events.append(row);
  }
}

async function submitBid(bid) {
  try {
    state = await postJson("/api/bid", { bid });
    render();
  } catch (error) {
    showActionError(error.message);
  }
}

async function submitPlay(card) {
  try {
    state = await postJson("/api/play", { suit: card.suit, rank: card.rank });
    render();
  } catch (error) {
    showActionError(error.message);
  }
}

async function nextRound() {
  try {
    state = await postJson("/api/next-round", {});
    render();
  } catch (error) {
    showActionError(error.message);
  }
}

async function updateProbabilityVisibility() {
  if (!state || !state.model_checkpoint) return;
  if (botAdvanceTimer) {
    window.clearTimeout(botAdvanceTimer);
    botAdvanceTimer = null;
  }
  liveProbabilitiesInput.disabled = true;
  try {
    state = await postJson("/api/probabilities", {
      visible: liveProbabilitiesInput.checked,
    });
    render();
  } catch (error) {
    liveProbabilitiesInput.checked = Boolean(state.show_probabilities);
    showActionError(error.message);
  } finally {
    liveProbabilitiesInput.disabled = !state.model_checkpoint;
    maybeScheduleBotAdvance();
  }
}

function maybeScheduleBotAdvance() {
  if (
    !state ||
    state.done ||
    state.round_over ||
    !["bidding", "playing"].includes(state.phase) ||
    state.human_turn ||
    advancingBot
  ) return;
  if (botAdvanceTimer) return;
  const delay = state.phase === "playing" ? CARD_PLAY_DELAY_MS : BID_DELAY_MS;
  botAdvanceTimer = window.setTimeout(advanceBot, delay);
}

function actionProbabilityForBid(bid) {
  if (state.model_action_probabilities?.phase !== "bidding") return null;
  return state.model_action_probabilities.actions.find((action) => action.bid === bid) || null;
}

function actionProbabilityForCard(key) {
  if (state.model_action_probabilities?.phase !== "playing") return null;
  return state.model_action_probabilities.actions.find((action) => action.card_key === key) || null;
}

function formatProbability(probability) {
  if (!Number.isFinite(probability)) return "—";
  return Number(probability).toPrecision(4);
}

function roundScorecardHtml(scores) {
  const rows = state.players
    .map((player) => `
      <div class="result-score-row">
        <span>${escapeHtml(player.name)}</span>
        <strong>+${scores[player.id] ?? 0}</strong>
        <span>${player.score} total</span>
      </div>
    `)
    .join("");
  return `<div class="result-scorecard">${rows}</div>`;
}

async function advanceBot() {
  botAdvanceTimer = null;
  if (!state || state.done || state.human_turn || advancingBot) return;
  advancingBot = true;
  try {
    state = await postJson("/api/advance", {});
    render();
  } catch (error) {
    showActionError(error.message);
  } finally {
    advancingBot = false;
    maybeScheduleBotAdvance();
  }
}

function pointLabel(pointProb) {
  return pointProb == null ? "—" : `${Math.round(pointProb * 100)}%`;
}

function suitPresenceHtml(suitPresence) {
  if (!suitPresence) return "";
  const symbols = { spades: "♠", hearts: "♥", diamonds: "♦", clubs: "♣" };
  const chips = Object.entries(symbols)
    .map(([suit, symbol]) => {
      const prob = suitPresence[suit];
      if (prob == null) return "";
      const pct = Math.round(prob * 100);
      const red = suit === "hearts" || suit === "diamonds" ? " red" : "";
      const faded = prob < 0.5 ? " faded" : "";
      return `<span class="suit-chip${red}${faded}">${symbol}${pct}</span>`;
    })
    .join("");
  return `<div class="suit-presence" title="Model belief: chance the player still holds each suit">${chips}</div>`;
}

function predictionHtml(player) {
  const prediction = player.prediction;
  if (!prediction?.rows?.length) return "";
  const self = prediction.rows.find((row) => row.player === player.id);
  const point = pointLabel(self?.bid_hit_prob);
  const rows = prediction.rows
    .map((row) => {
      const name = playerName(row.player);
      return `
        <div class="prediction-row">
          <span>${escapeHtml(name)}</span>
          <strong>Bid hit</strong>
          <span>${pointLabel(row.bid_hit_prob)}</span>
        </div>
        ${suitPresenceHtml(row.suit_presence)}
      `;
    })
    .join("");
  return `
    <div class="prediction-box">
      <div class="prediction-head">
        <span>Schema-v6 model POV</span>
        <strong>Your bid hit ${point}</strong>
      </div>
      <div class="prediction-grid">
        ${rows}
      </div>
    </div>
  `;
}

function cardHtml(card) {
  const isTrump = state?.trump && card.suit === state.trump;
  return `
    <div class="card ${card.color}${isTrump ? " trump-card" : ""}">
      <span>${escapeHtml(card.label)}</span>
      ${isTrump ? '<small>trump</small>' : ""}
    </div>
  `;
}

function eventText(event) {
  if (event.type === "bid") return `${playerName(event.player)} bid ${event.bid}`;
  if (event.type === "play") return `${playerName(event.player)} played ${event.card.label}`;
  if (event.type === "trick_win") return `${playerName(event.player)} won trick ${event.trick_index + 1}`;
  if (event.type === "round_start") return "Round started";
  if (event.type === "round_end") return "Round ended";
  return event.type;
}

function phaseTitle(gameState) {
  if (gameState.game_over) return "Game Over";
  if (gameState.round_over) return "Round Over";
  if (gameState.phase === "bidding") return "Bidding";
  if (gameState.phase === "playing") return "Playing";
  return gameState.phase;
}

function playerName(playerId) {
  if (playerId === null || playerId === undefined) return "Nobody";
  return playerId === 0 ? "You" : `Model ${playerId}`;
}

function trickOrder(leader, playerCount) {
  return Array.from({ length: playerCount }, (_, index) => (leader + index) % playerCount);
}

function suitTitle(suit) {
  return {
    spades: "Spades",
    hearts: "Hearts",
    diamonds: "Diamonds",
    clubs: "Clubs",
  }[suit] || suit;
}

function ordinal(value) {
  const suffix = value % 100 >= 10 && value % 100 <= 20 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[value % 10] || "th";
  return `${value}${suffix}`;
}

function showSetupError(message) {
  let error = setup.querySelector(".error");
  if (!error) {
    error = document.createElement("div");
    error.className = "error";
    form.prepend(error);
  }
  error.textContent = message;
}

function showActionError(message) {
  const root = document.querySelector("#action-panel");
  const error = document.createElement("div");
  error.className = "error";
  error.textContent = message;
  root.append(error);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

refreshSetupBounds();
refreshGameMode();
