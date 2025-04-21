document.getElementById("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const inputValue = document.getElementById("query").value;

    try {
        const response = await fetch("http://127.0.0.1:8000", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: inputValue })
        });
        
        const result = await response.json();
        const rawContent = result.response;
        console.log(response)
        let htmlContent = rawContent.replace(/```[\w]*\n?/, '').replace(/```$/, '');
        document.getElementById("report").innerHTML = htmlContent;
    } catch (error) {
        console.error("Error:", error);
    }
});
