function scrollCarousel(id, direction) {
    const container = document.getElementById(id);
    const scrollAmount = 200;
    container.scrollBy({
      left: direction * scrollAmount,
      behavior: 'smooth'
    });
  }
  <script>
  async function predictGenre() {
    const text = document.getElementById("movieText").value;
    const resultBox = document.getElementById("result");

    if (!text.trim()) {
      resultBox.innerText = "⚠️ Please enter a movie plot.";
      return;
    }

    resultBox.innerText = "⏳ Predicting...";

    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ plot: text })  // ✅ use 'plot' key to match app.py
    });

    const result = await response.json();
    resultBox.innerText = "🎬 Predicted Genres: " + result.genres.join(", ");
  }
</script>
