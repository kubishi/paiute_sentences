// on btn-translator click, get input-translator-english value and sent to /api/translator/translate
$(document).ready(function() {
    $('#input-translator-english').keypress(function(event) {
        if (event.which == 13) { // 13 is the keycode for the Enter key
            event.preventDefault(); // Prevent the default action (form submission)
            $('#btn-translator').click(); // Trigger the submit button click event
        }
    });

    $("#btn-translator").on("click", function() {
        // disable button and add a bootstrap spinner
        $("#btn-translator").prop("disabled", true);
        $("#btn-translator").html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...');
        
        $("#output-translator-english").text("");
        $("#output-translator-paiute").text("");
        $("#output-translator-warning").text("");
        $("#output-translator-message").text("");
        $("#translation-section").hide();
        $("#translation-warning").hide();
        $("#translation-message").hide();

        var english = $("#input-translator-english").val();
        fetch("/api/translator/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                english: english
            })
        }).then(res => {
            if (res.status == 200) {
                let res_json = res.json();
                return res_json;
            } else if (res.status == 429) {
                throw new Error(`Too many requests. Please try again later.`);
            } else{
                throw new Error(`Unknown error: ${res.status}`);
            }
        }).then(response => {
            console.log(response);
            $("#output-translator-english").text(response.english);
            $("#output-translator-paiute").text(response.paiute);
            if (response.warning) {
                $("#output-translator-warning").text(`*${response.warning}`);
                $("#translation-warning").show();
            }
            if (response.message) {
                $("#output-translator-message").text(`*${response.message}`);
                $("#translation-message").show();
            }
            $("#translation-section").show();
        }).catch(error => {
            console.log(error);
            $("#output-translator-warning").text(`*${error}`);
            $("#translation-warning").show();
        }).finally(() => {
            // re-enable button and remove bootstrap spinner
            $("#btn-translator").prop("disabled", false);
            $("#btn-translator").html("Translate");
        });
    });
    
});