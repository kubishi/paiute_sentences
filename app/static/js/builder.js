const dropdownIDLabels = {
    '#subject-noun': 'Subject',
    '#subject-suffix': 'Suffix',
    '#object-noun': 'Object',
    '#object-suffix': 'Suffix',
    '#object-pronoun': 'Object Pronoun',
    '#verb': 'Verb',
    '#verb-tense': 'Tense'
}

const dropdowns = {};
Object.keys(dropdownIDLabels).forEach((id, index) => {
    dropdowns[id] = new Choices(id, {
        itemSelectText: '',
        removeItems: true,
        removeItemButton: true,
        placeholder: true,
        placeholderValue: '',
        shouldSort: false,
        allowHTML: false,
    });
});

function setChoices(dropdownID, choices, value, requirement) {
    $(dropdownID).empty()
    // var element = document.getElementById(dropdownID);
    const choicesDropdown = dropdowns[dropdownID];

    if (choices.length == 0) {
        choicesDropdown.setChoices([{
            value: '',
            label: dropdownIDLabels[dropdownID]
        }], 'value', 'label', true);
    } else {
        // choicesDropdown.setChoices(choices.map(choice => {
        //     return {
        //         value: choice[0],
        //         label: choice[1]
        //     }
        // }), 'value', 'label', true);
        // include the placeholder
        choicesDropdown.setChoices([{
            value: '',
            label: dropdownIDLabels[dropdownID]
        }].concat(choices.map(choice => {
            return {
                value: choice[0],
                label: choice[1]
            }
        }
        )), 'value', 'label', true);
    }

    // get label for value from choices
    var label = value;
    for (var i = 0; i < choices.length; i++) {
        if (choices[i][0] == value) {
            label = choices[i][1]
            break
        }
    }

    // set the value
    if (value != null) {
        // choicesDropdown.setValue([{value: value, label: label}])
        choicesDropdown.setChoiceByValue(value)
    } else {
        // choicesDropdown.setValue([{value: null, label: dropdownIDLabels[dropdownID]}])
        choicesDropdown.setChoiceByValue('')
    }

    // set the requirement
    if (requirement == "required") {
        choicesDropdown.enable()
    } else if (requirement == "optional") {
        choicesDropdown.enable()
    } else if (requirement == "disabled") {
        choicesDropdown.disable()
    }
}

function doUpdate(res) {
    choices = res.choices
    console.log(choices);
    setChoices(
        '#subject-noun', 
        choices.subject_noun.choices, choices.subject_noun.value, choices.subject_noun.requirement
    )
    setChoices(
        '#subject-suffix',
        choices.subject_suffix.choices, choices.subject_suffix.value, choices.subject_suffix.requirement
    )
    setChoices(
        '#verb', 
        choices.verb.choices, choices.verb.value, choices.verb.requirement
    )
    setChoices(
        '#verb-tense', 
        choices.verb_tense.choices, choices.verb_tense.value, choices.verb_tense.requirement
    )
    setChoices(
        '#object-pronoun', 
        choices.object_pronoun.choices, choices.object_pronoun.value, choices.object_pronoun.requirement
    )
    setChoices(
        '#object-noun', 
        choices.object_noun.choices, choices.object_noun.value, choices.object_noun.requirement
    )
    setChoices(
        '#object-suffix', 
        choices.object_suffix.choices, choices.object_suffix.value, choices.object_suffix.requirement
    )

    // Update sentence if it exists
    if (res.sentence.length > 0) {
        formattedSentence = res.sentence.map(word => {
            if (word.type == "subject") {
                return `<span class="text-danger">${word.text}</span>`
            } else if (word.type == "verb") {
                return `<span class="text-primary">${word.text}</span>`
            } else if (word.type == "object") {
                return `<span class="text-success">${word.text}</span>`
            } else {
                return word.text
            }
        })

        // Set contents of #sentence to formattedSentence
        $('#sentence').html(formattedSentence.join(" "))
        // make btn-translate visible by removing display: none
        $('#btn-translate').css('display', '')
        // enable btn-translate
        $('#btn-translate').prop('disabled', false)
        // remove translation 
        $('#translation').html("")
        // set btn-randomize to invisible
        $('#btn-randomize').css('display', 'none')
    } else {
        $('#sentence').html("Your sentence isn't valid yet, please select more words.")
        // make btn-translate invisible
        $('#btn-translate').css('display', 'none')
        // remove translation 
        $('#translation').html("")
        // set btn-random to default display
        $('#btn-randomize').css('display', '')
    }
}

function updateUI(url) {
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": $('#subject-noun').val(),
            "subject_suffix": $('#subject-suffix').val(),
            "verb": $('#verb').val(),
            "verb_tense": $('#verb-tense').val(),
            "object_pronoun": $('#object-pronoun').val(),
            "object_noun": $('#object-noun').val(),
            "object_suffix": $('#object-suffix').val()
        })
    }).catch(err => {
        console.error(err);
    }).then(response => response.json()).then(res => doUpdate(res));
}

function startIntro() {
    fetch('/api/builder/random-example', {
        method: 'GET',
        headers: {'Content-Type': 'application/json'}
    }).catch(err => {
        console.error(err);
    }).then(response => response.json()).then(res => {
        doUpdate(res);

        const subjectNoun = $('#subject-noun').val()
        const subjectSuffix = $('#subject-suffix').val()
        const objectNoun = $('#object-noun').val()
        const objectSuffix = $('#object-suffix').val()
        const verb = $('#verb').val()
        const verbTense = $('#verb-tense').val()
        const objectPronoun = $('#object-pronoun').val()


        const steps = [
            {
                intro: "Welcome to the Owens Valley Paiute Sentence Builder. Let's build our first sentence together! Click 'Next' to continue."
            },
            {
                element: '#subject-noun-choice',
                intro: `The subject is the doer of the action in a sentence. In this example, we've selected the word ${subjectNoun}`
            },
            {
                element: '#subject-suffix-choice',
                intro: `The subject suffix tells us whether the subject is nearby/visible (-ii) or far away (-uu). In this example, we've selected the suffix ${subjectSuffix}`,
            },
            {
                element: '#object-noun-choice',
                intro: `The object is the receiver of the action in a sentence. In this example, we've selected the word ${objectNoun}`,
            },
            {
                element: '#object-suffix-choice',
                intro: `The object suffix tells us whether the object is nearby/visible (-eika) or far away (-oka). In this example, we've selected the suffix ${objectSuffix}`,
            },
            {
                element: '#verb-choice',
                intro: `The verb describes the action that the subject is taking. In this example, we've selected the word ${verb}`,
            },
            {
                element: '#verb-tense-choice',
                intro: `The verb tense indicates the time when the action is taking place. In this example, we've selected the tense ${verbTense}`,
            },
            {
                element: '#object-pronoun-choice',
                intro: `The object pronoun matches the object suffix (nearby or far away). In this example, we've selected the pronoun ${objectPronoun}`,
            },
            {
                element: '#sentence',
                intro: "Now that we've selected enough elements to create a sentence, our sentence has appeared here!",
            },
            {
                element: '#translation-section',
                intro: "Press the 'Translate' button to translate our sentence into English! The sentence might be very strange, like 'the dog will drink the house', and that's okay! The sentence might sound weird, but it's completely grammatically correct! Also, keep in mind that translation is a new feature and may not always be 100% accurate.",
            },
            {
                intro: "That's it! You can now create your own sentences in Owens Valley Paiute. You don't always need to select a value for each dropdown. For example, I am running (poyoha-ti nüü) has no object! Try out a bunch of sentences and see if you can make out the patterns!",
            }
        ];

        const driverSteps = steps.map(step => {
            return {
                element: step.element,
                popover: {
                    title: "",
                    description: step.intro,
                }
            }
        });

        Object.keys(dropdownIDLabels).forEach((id, index) => {
            dropdowns[id].disable()
        });

        const Driver = window.driver.js.driver;
        const driver = new Driver({
            className: 'custom-bottom-tooltip',
            animate: true,
            opacity: 0.75,
            padding: 10,
            allowClose: true,
            overlayClickNext: true,
            doneBtnText: 'Done', 
            closeBtnText: 'Close', 
            nextBtnText: 'Next', 
            prevBtnText: 'Previous',
            stageBackground: '#ffffff',
            steps: driverSteps,
            // disable buttons on start of tour
            onDestroyStarted: () => {
                // update UI to re-enable buttons
                updateUI('/api/builder/choices')
                driver.destroy();
            }
        });

        driver.drive();
    });
}

// implement translate button
$('#btn-translate').click(function() {
    // get the current values of all dropdowns
    subject_noun = $('#subject-noun').val()
    subject_suffix = $('#subject-suffix').val()
    verb = $('#verb').val()
    verb_tense = $('#verb-tense').val()
    object_pronoun = $('#object-pronoun').val()
    object_noun = $('#object-noun').val()
    object_suffix = $('#object-suffix').val()

    // disable btn-translate
    $('#btn-translate').prop('disabled', true)

    // call /api/builder/translate with the current values of all dropdowns
    fetch('/api/builder/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": subject_noun,
            "subject_suffix": subject_suffix,
            "verb": verb,
            "verb_tense": verb_tense,
            "object_pronoun": object_pronoun,
            "object_noun": object_noun,
            "object_suffix": object_suffix
        })
    }).catch(err => {
        console.log(`Status code: ${err.status}`);
        $('#btn-translate').prop('disabled', false);
    }).then(res => {
        if (res.status == 200) {
            let res_json = res.json();
            return res_json;
        } else if (res.status == 429) {
            throw new Error(`Too many requests. Please try again later.`);
        } else{
            throw new Error(`Unknown error: ${res.status}`);
        }
    }).then(res_json => {
        $('#translation').html(res_json.translation);
        $('#btn-translate').css('display', 'none');
    }).catch(err => {
        console.error(err);
        $('#translation').html(`<br/>${err.message}`);
        $('#btn-translate').prop('disabled', false);
    });
})

// call updateUI() when the page loads
$(document).ready(function() {    
    const list = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    list.map((el) => {
      let opts = {
        animation: false,
      }
      if (el.hasAttribute('data-bs-content-id')) {
        opts.content = document.getElementById(el.getAttribute('data-bs-content-id')).innerHTML;
        opts.html = true;
      }
      new bootstrap.Popover(el, opts);
    })

    updateUI('/api/builder/choices')
})
