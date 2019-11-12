// Override modal body to display a custom message suggesting a fix
function suggestNotebookShutdownOnErrorModal() {
    // Check to see if there is a modal dialog
    if ($('.modal-dialog').length != 0) {
        var title = $('.modal-title')[0].textContent
        if (title == "Kernel Restarting" || title == "Dead kernel") {
            // Append running notebooks message if absent
            var modalBody = $('.modal-dialog .modal-content .modal-body')[0];
            var runningNotebooksClassName = 'running-notebooks'
            if ($('.' + runningNotebooksClassName, modalBody).length == 0) {
                var p = document.createElement('p');
                p.className = runningNotebooksClassName;
                var runningNotebooksLink = document.createElement('a');
                var linkText = document.createTextNode("running notebooks");
                runningNotebooksLink.appendChild(linkText);
                runningNotebooksLink.title = "running notebooks";
                runningNotebooksLink.href = "/#running";
                p.appendChild(document.createTextNode(
                    "Kernel errors can occur when the notebook system runs out of memory, " + 
                    "often when users run multiple notebooks at once. " +
                    "Please check the list of "));
                p.appendChild(runningNotebooksLink);
                p.appendChild(document.createTextNode(
                    " and shut down any notebooks that you are not using."));
                modalBody.appendChild(p);
            }
        }
    }
}

window.setInterval(suggestNotebookShutdownOnErrorModal, 1000)
