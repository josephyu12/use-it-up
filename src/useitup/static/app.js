const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ═══════════════════════════════════════════════════
   Emoji Mappings
   ═══════════════════════════════════════════════════ */

const CATEGORY_EMOJI = {
  "Fresh produce": "🥬",
  "Herbs": "🌿",
  "Proteins": "🥩",
  "Dairy": "🧀",
  "Grains & starches": "🌾",
  "Oils, vinegars & spices": "🧂",
  "Condiments & pantry": "🫙",
};

const CONSTRAINT_EMOJI = {
  "vegan": "🌱",
  "vegetarian": "🥕",
  "gluten-free": "🌾",
  "dairy-free": "🥛",
  "nut-free": "🥜",
  "low-carb": "📉",
  "high-protein": "💪",
  "low-cost": "💰",
  "quick": "⚡",
};

const GOAL_EMOJI = {
  "high_protein": "💪",
  "low_cost": "💰",
  "vegetarian": "🥕",
  "vegan": "🌱",
  "low_carb": "📉",
  "quick": "⚡",
  "dairy_free": "🥛",
};

const CUISINE_EMOJI = {
  "Italian": "🇮🇹",
  "Mexican": "🇲🇽",
  "Asian": "🥢",
  "Indian": "🇮🇳",
  "American": "🇺🇸",
  "Mediterranean": "🫒",
  "French": "🇫🇷",
  "Middle Eastern": "🧆",
};

const DIFFICULTY_LABELS = {
  1: "Easy",
  2: "Moderate",
  3: "Intermediate",
  4: "Advanced",
  5: "Expert",
};

/* ═══════════════════════════════════════════════════
   Pantry Groups
   ═══════════════════════════════════════════════════ */

const PANTRY_GROUPS = [
  {
    label: "Fresh produce",
    items: [
      "garlic", "onion", "red onion", "shallot", "scallion",
      "tomato", "cherry tomatoes", "bell pepper", "red bell pepper", "jalapeño",
      "cucumber", "avocado", "lemon", "lime", "ginger",
      "carrot", "celery", "mushroom", "zucchini", "potato", "sweet potato",
      "spinach", "kale", "arugula", "broccoli", "cauliflower",
      "eggplant", "asparagus", "corn", "cabbage", "bok choy",
      "peas", "green beans", "lettuce", "radish",
    ],
  },
  {
    label: "Herbs",
    items: [
      "basil", "cilantro", "parsley", "mint", "rosemary",
      "thyme", "oregano", "dill", "sage", "chives", "tarragon",
    ],
  },
  {
    label: "Proteins",
    items: [
      "chicken breast", "chicken thighs", "ground beef", "ground turkey",
      "bacon", "salmon", "shrimp", "canned tuna",
      "tofu", "tempeh", "eggs",
      "black beans", "chickpeas", "cannellini beans", "lentils",
      "pork chop", "sausage", "lamb", "cod",
      "pinto beans", "kidney beans",
    ],
  },
  {
    label: "Dairy",
    items: [
      "butter", "milk", "heavy cream", "greek yogurt",
      "parmesan", "mozzarella", "feta cheese", "cheddar cheese", "cream cheese",
      "sour cream", "ricotta", "goat cheese", "brie", "ghee",
    ],
  },
  {
    label: "Grains & starches",
    items: [
      "spaghetti", "penne", "fettuccine", "brown rice", "white rice", "quinoa",
      "bread", "corn tortillas", "flour tortillas", "flour",
      "oats", "ramen noodles", "couscous", "rice noodles",
      "naan", "pita bread", "breadcrumbs", "polenta",
    ],
  },
  {
    label: "Oils, vinegars & spices",
    items: [
      "olive oil", "vegetable oil", "sesame oil", "coconut oil",
      "balsamic vinegar", "rice vinegar", "apple cider vinegar",
      "salt", "black pepper", "red pepper flakes",
      "cumin", "paprika", "cinnamon", "turmeric", "curry powder",
      "chili powder", "cayenne", "garam masala", "coriander",
      "smoked paprika", "nutmeg", "bay leaf",
    ],
  },
  {
    label: "Condiments & pantry",
    items: [
      "soy sauce", "fish sauce", "hot sauce", "dijon mustard", "mayonnaise",
      "honey", "maple syrup", "peanut butter", "tahini",
      "canned tomatoes", "tomato paste", "tomato sauce", "salsa", "coconut milk",
      "chicken broth", "vegetable broth",
      "miso paste", "worcestershire", "sriracha", "hoisin",
      "capers", "olives", "nutritional yeast", "sugar", "brown sugar",
    ],
  },
];

const KNOWN_PANTRY_ITEMS = new Set(PANTRY_GROUPS.flatMap((group) => group.items));
const STEPS = ["pantry", "constraints", "goals", "taste"];

/* ═══════════════════════════════════════════════════
   Templates
   ═══════════════════════════════════════════════════ */

const TEMPLATES = [
  {
    emoji: "🍛",
    label: "Indian curry night",
    sub: "Chickpeas · coconut · spice",
    pantry: ["chickpeas", "lentils", "coconut milk", "canned tomatoes",
             "onion", "garlic", "ginger", "cumin", "turmeric",
             "curry powder", "cinnamon", "cilantro", "white rice", "butter",
             "greek yogurt"],
    hard_constraints: [],
    goals: ["vegetarian"],
    preferred_cuisines: ["Indian"],
    max_prep: 45,
  },
  {
    emoji: "🌮",
    label: "Mexican pantry check",
    sub: "Tortillas · beans · lime",
    pantry: [
      "garlic", "onion", "red onion", "shallot", "tomato", "cherry tomatoes",
      "cucumber", "avocado", "lemon", "lime", "ginger", "mushroom", "spinach",
      "arugula", "eggplant", "corn", "cabbage", "lettuce",
      "cilantro", "parsley", "rosemary", "thyme",
      "chicken breast", "chicken thighs", "ground beef", "ground turkey", "tofu",
      "eggs", "black beans", "chickpeas", "lentils", "kidney beans",
      "butter", "milk", "greek yogurt", "feta cheese", "cheddar cheese",
      "brown rice", "white rice", "quinoa", "corn tortillas", "flour tortillas", "ramen noodles",
      "olive oil", "sesame oil", "balsamic vinegar", "red pepper flakes",
      "cumin", "paprika", "cinnamon", "turmeric",
      "soy sauce", "fish sauce", "canned tomatoes", "coconut milk"
    ],
    hard_constraints: [],
    goals: ["low_cost"],
    preferred_cuisines: ["Mexican"],
    max_prep: 45,
  },
];

/* ═══════════════════════════════════════════════════
   State
   ═══════════════════════════════════════════════════ */

const state = {
  currentStep: 0,
  pantry: [],
  selectedPantry: new Set(),
  constraints: new Set(),
  goals: new Set(),
  cuisines: new Set(),
  ratingHistory: [],
  allRecipes: [],
  libraryQuery: "",
  activeModalStep: -1,
  lastRecommendData: null,
};

/* ═══════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════ */

async function init() {
  const [vocab, demo, recipes] = await Promise.all([
    fetch("/api/vocabulary").then((r) => r.json()),
    fetch("/api/profile/demo").then((r) => r.json()),
    fetch("/api/recipes").then((r) => r.json()),
  ]);

  state.demoProfile = demo;
  state.allRecipes = recipes;

  renderPillGroup("#constraints", vocab.dietary_tags, state.constraints, () => updateCounts(), labelConstraint);
  renderPillGroup("#goals", vocab.goals, state.goals, () => updateCounts(), labelGoal);
  renderPillGroup("#cuisines", vocab.cuisines, state.cuisines, () => {}, labelCuisine);
  renderTemplates();
  renderPantryBoard();
  renderLibrary();
  updateCounts();
  updateStepUI();

  $("#template-toggle").addEventListener("click", () => {
    $("#templates").classList.toggle("hidden");
  });
  $("#demo-btn").addEventListener("click", async () => {
    loadProfile(state.demoProfile);
    await runRecommend();
  });
  $("#pantry").addEventListener("input", syncPantryState);
  $("#max-prep").addEventListener("input", (e) => {
    $("#max-prep-val").textContent = `${e.target.value} min`;
  });
  $("#submit-btn").addEventListener("click", runRecommend);
  $("#next-pantry-btn").addEventListener("click", () => setStep(1));
  $("#back-constraints-btn").addEventListener("click", () => setStep(0));
  $("#next-constraints-btn").addEventListener("click", () => setStep(2));
  $("#back-goals-btn").addEventListener("click", () => setStep(1));
  $("#next-goals-btn").addEventListener("click", () => setStep(3));
  $("#back-taste-btn").addEventListener("click", () => setStep(2));
  $("#library-search").addEventListener("input", (e) => {
    state.libraryQuery = e.target.value.trim().toLowerCase();
    renderLibrary();
  });

  // Lazy-render library only when the details panel is opened
  const libraryDetails = document.querySelector(".advanced-panel:last-of-type");
  if (libraryDetails) {
    libraryDetails.addEventListener("toggle", () => {
      if (libraryDetails.open) renderLibrary();
    });
  }

  // Global escape to close modals
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeRecipeModal();
  });
}

/* ═══════════════════════════════════════════════════
   Pill Group Rendering (with emoji support)
   ═══════════════════════════════════════════════════ */

function renderPillGroup(sel, items, store, onChange = () => {}, labelFn = (x) => x) {
  const host = $(sel);
  host.innerHTML = "";
  items.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pill";
    btn.dataset.value = item;
    btn.innerHTML = labelFn(item);
    btn.addEventListener("click", () => {
      if (store.has(item)) {
        store.delete(item);
        btn.classList.remove("active");
      } else {
        store.add(item);
        btn.classList.add("active");
      }
      onChange();
    });
    host.appendChild(btn);
  });
}

function labelConstraint(key) {
  const emoji = CONSTRAINT_EMOJI[key] || "";
  const text = key.replace(/-/g, " ").replace(/_/g, " ");
  return emoji ? `<span class="pill-emoji">${emoji}</span>${escapeHtml(text)}` : escapeHtml(text);
}

function labelGoal(key) {
  const emoji = GOAL_EMOJI[key] || "";
  const text = key.replace(/_/g, " ");
  return emoji ? `<span class="pill-emoji">${emoji}</span>${escapeHtml(text)}` : escapeHtml(text);
}

function labelCuisine(key) {
  const emoji = CUISINE_EMOJI[key] || "🍽️";
  return `<span class="pill-emoji">${emoji}</span>${escapeHtml(key)}`;
}

/* ═══════════════════════════════════════════════════
   Templates
   ═══════════════════════════════════════════════════ */

function renderTemplates() {
  const host = $("#templates");
  host.innerHTML = "";
  TEMPLATES.forEach((template) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "template-card";
    btn.innerHTML = `
      <span class="emoji">${template.emoji}</span>
      <span>
        <strong>${escapeHtml(template.label)}</strong>
        <span>${escapeHtml(template.sub)}</span>
      </span>
    `;
    btn.addEventListener("click", () => {
      applyTemplate(template);
      setStep(0);
    });
    host.appendChild(btn);
  });
}

/* ═══════════════════════════════════════════════════
   Pantry Board with Category Emojis
   ═══════════════════════════════════════════════════ */

function renderPantryBoard() {
  const host = $("#pantry-board");
  host.innerHTML = "";
  PANTRY_GROUPS.forEach((group) => {
    const card = document.createElement("section");
    card.className = "pantry-group";
    const emoji = CATEGORY_EMOJI[group.label] || "";
    card.innerHTML = `<h3><span class="cat-emoji">${emoji}</span> ${group.label}</h3>`;
    const grid = document.createElement("div");
    grid.className = "pantry-grid";
    group.items.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pantry-item";
      btn.dataset.item = item;
      btn.innerHTML = `<span class="dot">+</span><span>${escapeHtml(item)}</span>`;
      btn.addEventListener("click", () => {
        if (state.selectedPantry.has(item)) {
          state.selectedPantry.delete(item);
        } else {
          state.selectedPantry.add(item);
        }
        syncPantryBoard();
        syncPantryState();
      });
      grid.appendChild(btn);
    });
    card.appendChild(grid);
    host.appendChild(card);
  });
  syncPantryBoard();
}

function syncPantryBoard() {
  $$(".pantry-item").forEach((btn) => {
    const active = state.selectedPantry.has(btn.dataset.item);
    btn.classList.toggle("active", active);
    btn.querySelector(".dot").textContent = active ? "✓" : "+";
  });
}

function syncPantryState() {
  const extras = $("#pantry").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  state.pantry = Array.from(new Set([...state.selectedPantry, ...extras]));
  updateCounts();
}

function updateCounts() {
  $("#pantry-count").textContent = `${state.pantry.length} item${state.pantry.length === 1 ? "" : "s"} selected`;
  $("#constraint-count").textContent = state.constraints.size
    ? `${state.constraints.size} restriction${state.constraints.size === 1 ? "" : "s"} selected`
    : "No restrictions selected";
  $("#goal-count").textContent = state.goals.size
    ? `${state.goals.size} goal${state.goals.size === 1 ? "" : "s"} selected`
    : "No goals selected";
}

/* ═══════════════════════════════════════════════════
   Template / Profile Loading
   ═══════════════════════════════════════════════════ */

function applyTemplate(template) {
  const known = [];
  const extras = [];
  template.pantry.forEach((item) => {
    if (KNOWN_PANTRY_ITEMS.has(item)) {
      known.push(item);
    } else {
      extras.push(item);
    }
  });
  state.selectedPantry = new Set(known);
  $("#pantry").value = extras.join(", ");
  setPillState("#constraints", state.constraints, template.hard_constraints);
  setPillState("#goals", state.goals, template.goals);
  setPillState("#cuisines", state.cuisines, template.preferred_cuisines);
  $("#max-prep").value = template.max_prep;
  $("#max-prep-val").textContent = `${template.max_prep} min`;
  state.ratingHistory = [];
  syncPantryBoard();
  syncPantryState();
}

function loadProfile(profile) {
  const known = [];
  const extras = [];
  (profile.pantry || []).forEach((item) => {
    if (KNOWN_PANTRY_ITEMS.has(item)) {
      known.push(item);
    } else {
      extras.push(item);
    }
  });
  state.selectedPantry = new Set(known);
  $("#pantry").value = extras.join(", ");
  setPillState("#constraints", state.constraints, profile.hard_constraints || []);
  setPillState("#goals", state.goals, profile.soft_preferences?.goals || []);
  setPillState("#cuisines", state.cuisines, profile.soft_preferences?.preferred_cuisines || []);
  $("#max-prep").value = profile.soft_preferences?.max_prep_time_min || 45;
  $("#max-prep-val").textContent = `${$("#max-prep").value} min`;
  state.ratingHistory = profile.rating_history || [];
  syncPantryBoard();
  syncPantryState();
}

function setPillState(sel, store, values) {
  store.clear();
  values.forEach((value) => store.add(value));
  $$(`${sel} .pill`).forEach((btn) => {
    btn.classList.toggle("active", store.has(btn.dataset.value));
  });
  updateCounts();
}

/* ═══════════════════════════════════════════════════
   Wizard Navigation
   ═══════════════════════════════════════════════════ */

function setStep(index) {
  state.currentStep = Math.max(0, Math.min(index, STEPS.length - 1));
  updateStepUI();
}

function updateStepUI() {
  $$(".wizard-step").forEach((step, index) => {
    step.classList.toggle("hidden", index !== state.currentStep);
    step.classList.toggle("active", index === state.currentStep);
  });
  $("#step-counter").textContent = `Step ${state.currentStep + 1} of ${STEPS.length}`;
  const rail = $("#progress-rail");
  rail.innerHTML = "";
  STEPS.forEach((_, index) => {
    const seg = document.createElement("span");
    seg.className = "seg";
    if (index < state.currentStep) seg.classList.add("filled");
    if (index === state.currentStep) seg.classList.add("active");
    rail.appendChild(seg);
  });
}

/* ═══════════════════════════════════════════════════
   Recommend API Call
   ═══════════════════════════════════════════════════ */

async function runRecommend() {
  syncPantryState();
  $("#loading").classList.remove("hidden");
  $("#error-panel").classList.add("hidden");

  const body = {
    pantry: state.pantry,
    hard_constraints: Array.from(state.constraints),
    goals: Array.from(state.goals),
    preferred_cuisines: Array.from(state.cuisines),
    max_prep_time_min: parseInt($("#max-prep").value, 10),
    top_k: parseInt($("#top-k").value, 10),
    rating_history: state.ratingHistory,
  };

  try {
    const resp = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) {
      showError(data.detail || "Something went wrong.");
      return;
    }
    if (data.error) {
      showError(data.error);
      return;
    }
    state.lastRecommendData = data;
    renderResults(data);
  } catch (err) {
    showError(String(err));
  } finally {
    $("#loading").classList.add("hidden");
  }
}

/* ═══════════════════════════════════════════════════
   Results Rendering
   ═══════════════════════════════════════════════════ */

function renderResults(data) {
  $("#setup-view").classList.add("hidden");
  $("#results-view").classList.remove("hidden");
  $("#error-panel").classList.add("hidden");

  const summary = data.summary || {};
  $("#results-summary").innerHTML = `
    <div>
      <div class="eyebrow">🎉 Results</div>
      <h2>${data.recommendations.length} recipe${data.recommendations.length === 1 ? "" : "s"} for you</h2>
    </div>
    <div class="summary-pills">
      <span class="summary-pill">📊 Candidates <strong>${summary.candidates_considered ?? 0}</strong></span>
      <span class="summary-pill">✅ Passed <strong>${summary.survivors ?? 0}</strong></span>
      <span class="summary-pill">🎯 Core match <strong>${Math.round((summary.essential_coverage_top ?? 0) * 100)}%</strong></span>
    </div>
    <button type="button" class="btn-secondary" id="edit-setup-btn">← Edit setup</button>
  `;
  $("#edit-setup-btn").addEventListener("click", () => {
    $("#results-view").classList.add("hidden");
    $("#setup-view").classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  renderShoppingAlert(summary.shopping_advice);

  const stack = $("#recipe-stack");
  stack.innerHTML = "";
  data.recommendations.forEach((rec, idx) => {
    const card = renderRecipeCard(rec, idx);
    // Staggered entrance animation
    card.style.animationDelay = `${idx * 80}ms`;
    card.classList.add("stagger-in");
    stack.appendChild(card);
  });
  $("#decision-log-wrap").innerHTML = renderDecisionLog(data.decision_log);
  state.libraryRendered = false; // reset so library re-renders if opened
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showError(msg) {
  $("#setup-view").classList.add("hidden");
  $("#results-view").classList.remove("hidden");
  $("#results-summary").innerHTML = "";
  $("#shopping-alert").innerHTML = "";
  $("#recipe-stack").innerHTML = "";
  $("#decision-log-wrap").innerHTML = "";
  const panel = $("#error-panel");
  panel.classList.remove("hidden");
  panel.textContent = msg;
}

function renderShoppingAlert(advice) {
  const host = $("#shopping-alert");
  if (!advice?.show_alert) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = `
    <section class="shopping-alert">
      <div class="eyebrow">🛒 Shopping tip</div>
      <h3>Shopping trip recommended</h3>
      <p>${escapeHtml(advice.message || "A few staples would unlock stronger recipe matches.")}</p>
      <div class="shopping-suggestions">
        ${(advice.suggestions || []).map((item) => `<span class="shop-chip">+ ${escapeHtml(item)}</span>`).join("")}
      </div>
    </section>
  `;
}

/* ═══════════════════════════════════════════════════
   Recipe Card (with hero image, badges, click hint)
   ═══════════════════════════════════════════════════ */

function renderRecipeCard(rec, idx) {
  const recipe = rec.recipe;
  const essentialMissing = new Set(rec.match?.essential_missing_ingredients || []);
  const matchedIngredients = new Set((rec.match?.matched_ingredients || []).map((item) => item.toLowerCase()));
  const ingredients = recipe.ingredients.slice(0, 6).map((ingredient) => {
    const have = matchedIngredients.has(ingredient.name.toLowerCase());
    const missingCore = essentialMissing.has(ingredient.name.toLowerCase());
    return `<li class="${have ? "have" : "missing"}${missingCore ? " missing-core" : ""}">
      <span class="check">${have ? "✓" : missingCore ? "!" : "○"}</span>
      <span>${escapeHtml(ingredient.name)}</span>
    </li>`;
  }).join("");
  const moreCount = Math.max(0, recipe.ingredients.length - 6);

  // Build meta badges
  const cuisineEmoji = CUISINE_EMOJI[recipe.cuisine] || "🍽️";
  const diffLabel = DIFFICULTY_LABELS[recipe.difficulty] || `Level ${recipe.difficulty}`;
  const dietaryBadges = (recipe.dietary_tags || []).slice(0, 3).map((tag) => {
    const emoji = CONSTRAINT_EMOJI[tag] || "";
    return `<span class="meta-badge">${emoji ? `<span class="badge-emoji">${emoji}</span>` : ""}<span>${escapeHtml(tag)}</span></span>`;
  }).join("");

  const recipeMetaLine = `
    <span class="recipe-inline-meta-item">
      <span class="meta-icon" aria-hidden="true">${cuisineEmoji}</span>
      <span>${escapeHtml(recipe.cuisine)}</span>
    </span>
    <span class="recipe-inline-meta-sep" aria-hidden="true">·</span>
    <span class="recipe-inline-meta-item">
      <span class="meta-icon" aria-hidden="true">⏱️</span>
      <span>${recipe.total_time_min} min</span>
    </span>
    <span class="recipe-inline-meta-sep" aria-hidden="true">·</span>
    <span class="recipe-inline-meta-item">
      <span>${diffLabel}</span>
    </span>
  `;

  const card = document.createElement("article");
  card.className = "recipe-card";
  card.innerHTML = `
    <img
      class="recipe-hero-img"
      src="${recipe.image_url}"
      alt="${escapeHtml(recipe.name)}"
      loading="lazy"
      onerror="this.style.display='none'"
    />
    <div class="recipe-card-body">
      <div class="recipe-header">
        <div>
          <div class="eyebrow">${idx === 0 ? "⭐ Top pick" : `Option ${idx + 1}`}</div>
          <h3>${escapeHtml(recipe.name)}</h3>
          <p class="recipe-inline-meta">${recipeMetaLine}</p>
        </div>
        <div class="summary-pills">
          <span class="summary-pill"><span class="pill-icon" aria-hidden="true">🎯</span><span>Core <strong>${Math.round((rec.match?.essential_coverage ?? 0) * 100)}%</strong></span></span>
          <span class="summary-pill"><span class="pill-icon" aria-hidden="true">📊</span><span>Weighted <strong>${Math.round((rec.match?.weighted_coverage ?? 0) * 100)}%</strong></span></span>
        </div>
      </div>
      <div class="recipe-meta">
        ${dietaryBadges}
      </div>
      <div class="recipe-columns">
        <section>
          <h4><span class="section-emoji" aria-hidden="true">🥘</span><span>Ingredients</span></h4>
          <ul class="ingredient-list">${ingredients}</ul>
          ${moreCount > 0 ? `<p style="margin:8px 0 0;color:var(--ink-dim);font-size:13px">+${moreCount} more ingredient${moreCount === 1 ? "" : "s"}</p>` : ""}
        </section>
        <section>
          <h4><span class="section-emoji" aria-hidden="true">💡</span><span>Why this works</span></h4>
          <div class="explanation-block">${markdownLite(rec.explanation.goal_trace)}</div>
        </section>
      </div>
      <div class="recipe-click-hint">
        <span class="hint-icon" aria-hidden="true">👆</span>
        <span>Click to view full recipe & step-by-step instructions</span>
      </div>
    </div>
  `;

  card.addEventListener("click", () => openRecipeModal(rec));
  return card;
}

/* ═══════════════════════════════════════════════════
   Recipe Detail Modal
   ═══════════════════════════════════════════════════ */

function openRecipeModal(rec) {
  const recipe = rec.recipe;
  state.activeModalStep = 0;

  // Scroll-lock body
  document.body.style.overflow = "hidden";

  const essentialMissing = new Set(rec.match?.essential_missing_ingredients || []);
  const matchedIngredients = new Set((rec.match?.matched_ingredients || []).map((item) => item.toLowerCase()));

  const ingredientItems = recipe.ingredients.map((ingredient) => {
    const have = matchedIngredients.has(ingredient.name.toLowerCase());
    const missingCore = essentialMissing.has(ingredient.name.toLowerCase());
    const qty = ingredient.quantity ? `${ingredient.quantity}${ingredient.unit ? ` ${ingredient.unit}` : ""}` : "";
    return `<li class="${have ? "have" : "missing"}${missingCore ? " missing-core" : ""}">
      <span class="check">${have ? "✓" : missingCore ? "!" : "○"}</span>
      <span>${escapeHtml(ingredient.name)}</span>
      <span class="qty">${escapeHtml(qty)}</span>
    </li>`;
  }).join("");

  const cuisineEmoji = CUISINE_EMOJI[recipe.cuisine] || "🍽️";
  const diffLabel = DIFFICULTY_LABELS[recipe.difficulty] || `Level ${recipe.difficulty}`;
  const recipeMetaLine = `
    <span class="recipe-inline-meta-item">
      <span class="meta-icon" aria-hidden="true">${cuisineEmoji}</span>
      <span>${escapeHtml(recipe.cuisine)}</span>
    </span>
    <span class="recipe-inline-meta-sep" aria-hidden="true">·</span>
    <span class="recipe-inline-meta-item">
      <span class="meta-icon" aria-hidden="true">⏱️</span>
      <span>${recipe.total_time_min} min</span>
    </span>
    <span class="recipe-inline-meta-sep" aria-hidden="true">·</span>
    <span class="recipe-inline-meta-item">
      <span>${diffLabel}</span>
    </span>
  `;

  // Build dietary + flavor badges
  const allTags = [
    ...(recipe.dietary_tags || []).map((t) => ({ label: t, emoji: CONSTRAINT_EMOJI[t] || "" })),
    ...(recipe.flavor_profile || []).map((f) => ({ label: f, emoji: "" })),
  ];
  const metaBadges = allTags.map((t) =>
    `<span class="meta-badge">${t.emoji ? `<span class="badge-emoji">${t.emoji}</span>` : ""}<span>${escapeHtml(t.label)}</span></span>`
  ).join("");

  // Build instructions steps
  const instructions = recipe.instructions || [];
  const stepsHtml = instructions.map((step, i) =>
    `<div class="step-card${i === 0 ? " active-step" : ""}" data-step-idx="${i}">
      <span class="step-number">${i + 1}</span>
      <span class="step-text">${escapeHtml(step)}</span>
    </div>`
  ).join("");

  // Build explanation accordions
  const explanationSections = [];
  if (rec.explanation?.goal_trace) {
    explanationSections.push({
      title: "🎯 Goal Trace",
      body: markdownLite(rec.explanation.goal_trace),
    });
  }
  if (rec.explanation?.counterfactual) {
    explanationSections.push({
      title: "🔄 Counterfactual",
      body: markdownLite(rec.explanation.counterfactual),
    });
  }
  if (rec.explanation?.cbr_trace) {
    explanationSections.push({
      title: "🧠 CBR Trace",
      body: markdownLite(rec.explanation.cbr_trace),
    });
  }
  if (rec.explanation?.ingredient_utilization_report) {
    explanationSections.push({
      title: "📦 Ingredient Utilization",
      body: markdownLite(rec.explanation.ingredient_utilization_report),
    });
  }

  const explanationHtml = explanationSections.map((section) =>
    `<details class="explanation-accordion">
      <summary>${section.title}</summary>
      <div class="accordion-body">${section.body}</div>
    </details>`
  ).join("");

  // Nutrition summary
  const nutrition = recipe.nutrition || {};
  const nutritionBadges = Object.entries(nutrition)
    .filter(([, v]) => v > 0)
    .slice(0, 4)
    .map(([k, v]) => `<span class="meta-badge">${escapeHtml(k.replace(/_/g, " "))}: <strong>${Math.round(v)}</strong></span>`)
    .join("");

  const root = $("#recipe-modal-root");
  root.innerHTML = `
    <div class="recipe-modal-overlay" id="recipe-modal-overlay">
      <div class="recipe-modal" id="recipe-modal">
        <div class="modal-hero">
          <img src="${recipe.image_url}" alt="${escapeHtml(recipe.name)}" onerror="this.parentElement.style.display='none'" />
          <div class="modal-hero-gradient"></div>
          <div class="modal-hero-title">
            <h2>${escapeHtml(recipe.name)}</h2>
            <p class="recipe-inline-meta">${recipeMetaLine}</p>
          </div>
          <button class="modal-close-btn" id="modal-close-btn" aria-label="Close">✕</button>
        </div>
        <div class="modal-body">
          <div class="modal-meta">
            ${metaBadges}
            ${nutritionBadges}
          </div>

          <div class="modal-ingredients">
            <h3 class="modal-section-title"><span class="section-emoji">🥘</span> Ingredients</h3>
            <div class="modal-ingredient-grid">
              <ul class="ingredient-list">${ingredientItems}</ul>
            </div>
          </div>

          ${instructions.length > 0 ? `
          <div>
            <h3 class="modal-section-title"><span class="section-emoji">👨‍🍳</span> Step-by-step instructions</h3>
            <div class="steps-container" id="steps-container">
              ${stepsHtml}
            </div>
          </div>
          ` : ""}

          ${explanationHtml ? `
          <div class="modal-explanations">
            <h3 class="modal-section-title"><span class="section-emoji">🧠</span> Why this recipe?</h3>
            ${explanationHtml}
          </div>
          ` : ""}
        </div>
      </div>
    </div>
  `;

  // Event: close button
  $("#modal-close-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    closeRecipeModal();
  });

  // Event: clicking overlay background closes modal
  $("#recipe-modal-overlay").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeRecipeModal();
  });

  // Prevent card click from bubbling into overlay close
  $("#recipe-modal").addEventListener("click", (e) => {
    e.stopPropagation();
  });

  // Event: step cards — click to focus / zoom
  $$(".step-card").forEach((card) => {
    card.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(card.dataset.stepIdx, 10);
      state.activeModalStep = idx;
      $$(".step-card").forEach((c, i) => {
        c.classList.toggle("active-step", i === idx);
      });
    });
  });
}

function closeRecipeModal() {
  const root = $("#recipe-modal-root");
  if (!root.innerHTML) return;
  document.body.style.overflow = "";
  root.innerHTML = "";
  state.activeModalStep = -1;
}

/* ═══════════════════════════════════════════════════
   Library (with thumbnails)
   ═══════════════════════════════════════════════════ */

function renderLibrary() {
  const recipes = state.allRecipes.filter((recipe) => {
    if (!state.libraryQuery) return true;
    const haystack = [recipe.name, recipe.cuisine, ...(recipe.ingredients || []).map((ing) => ing.name)].join(" ").toLowerCase();
    return haystack.includes(state.libraryQuery);
  });
  $("#library-count").textContent = `${recipes.length} recipes`;
  const grid = $("#library-grid");
  // Only render first 30 for performance (images are heavy)
  const toShow = recipes.slice(0, 30);
  grid.innerHTML = toShow.map((recipe, i) => {
    const cuisineEmoji = CUISINE_EMOJI[recipe.cuisine] || "🍽️";
    return `
    <article class="library-card" data-recipe-id="${escapeHtml(recipe.id)}">
      <img
        class="library-card-img"
        src="${recipe.image_url}"
        alt="${escapeHtml(recipe.name)}"
        loading="lazy"
        onerror="this.style.display='none'"
      />
      <div class="library-card-body">
        <h3>${escapeHtml(recipe.name)}</h3>
        <p>${cuisineEmoji} ${escapeHtml(recipe.cuisine)} · ${recipe.total_time_min} min</p>
      </div>
    </article>
  `;
  }).join("");

  // Use event delegation instead of per-card listeners
  grid.onclick = (e) => {
    const card = e.target.closest(".library-card");
    if (!card) return;
    const recipeId = card.dataset.recipeId;
    const recipe = state.allRecipes.find((r) => r.id === recipeId);
    if (recipe) {
      openRecipeModal({
        recipe,
        match: null,
        explanation: {},
        cbr: {},
      });
    }
  };
  state.libraryRendered = true;
}

/* ═══════════════════════════════════════════════════
   Decision Log
   ═══════════════════════════════════════════════════ */

function renderDecisionLog(log) {
  if (!log?.length) return "";
  return `
    <div class="decision-log">
      ${log.map((entry) => `
        <div class="decision-row">
          <strong>${escapeHtml(entry.rule_name)}</strong>
          <span>${escapeHtml(entry.recipe_id)}</span>
          <span class="${entry.passed ? "pass" : "fail"}">${entry.passed ? "✅ PASS" : "❌ FAIL"}</span>
          <p>${escapeHtml(entry.reason)}</p>
        </div>
      `).join("")}
    </div>
  `;
}

/* ═══════════════════════════════════════════════════
   Utilities
   ═══════════════════════════════════════════════════ */

function escapeHtml(text) {
  if (text == null) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function markdownLite(md) {
  if (!md) return "";
  return md
    .split("\n")
    .filter((line) => line.trim() && !line.startsWith("## "))
    .map((line) => {
      if (line.startsWith("- ")) return `<p>• ${inline(line.slice(2))}</p>`;
      return `<p>${inline(line)}</p>`;
    })
    .join("");
}

function inline(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/_(.+?)_/g, "<em>$1</em>");
}

init();
