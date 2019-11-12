define(['require'], function(require) {

  var submit_notebook_button_click = function() {
    var coursera_metadata = IPython.notebook.metadata.coursera;
    var course_slug = coursera_metadata.course_slug;
    var graded_item_id = coursera_metadata.graded_item_id;
    var part_id = coursera_metadata.part_id;
    var submission_page_url = "https://www.coursera.org/learn/" + course_slug + "/programming/" + graded_item_id;
    var with_part_id = (part_id == null) ? '': '~' + part_id;
    var schema_name = course_slug + '~' + graded_item_id + with_part_id;

    require(['jquery', 'base/js/dialog'], function($, dialog) {
      var body = $('<div/>')
        .append($('<span> Go to </span>'))
        .append($('<a></a>')
          .text(submission_page_url)
          .attr('href', submission_page_url))
        .append($('<span> to see your submissions.</span>'))
        .append($('</br>'))
        .append($('</br>'))
        .append($('<span>Please note that submitting work that isn’t your own may result in permanent failure of this course or deactivation of your Coursera account.</span>'))
        .append($('</br>'))
        .append($('</br>'))
        .append($('<span> Submitting ... </span>'));

      dialog.modal({
        title: 'Submission Status',
        body: body,
        buttons: {
          'Ok': {}
        }
      });
      IPython.notebook.save_notebook().then(function() {
        execute_submission(function(out) {
          if ((out.content.name === "stdout" && out.content.text.trim() !== "")) {
            body.children().last().text(out.content.text);
          }
        }, schema_name);
      });
    });
  };

  var read_submission_token_from_cookie = function() {
    var value = document.cookie.replace(/(?:(?:^|.*;\s*)COURSERA_SUBMISSION_TOKEN\s*\=\s*([^;]*).*$)|^.*$/, "$1");
    return value ? value : '';
  };

  var execute_submission = function(handle_result, schema_name) {
    var callbacks = { 'iopub' : {'output' : handle_result}}
    var submission_token = read_submission_token_from_cookie();
    var expression =
      '%%python \n' +
      'from coursera.grading import submit \n' +
      'print(submit("' + submission_token + '", "' + schema_name + '"))';
    console.log(expression);
    IPython.notebook.kernel.execute(expression, callbacks, {silent: false});
  };

  var add_submit_button = function() {
    require(['jquery'], function ($) {
      var group_id = 'submit-notebook-button-group';
      var button_id = 'submit-notebook-button';
      IPython.toolbar.add_buttons_group([
        {
          'id'      : button_id,
          'label'   : 'Submit this notebook for grading.',
          'icon'    : 'fa-inbox', // overridden below
          'callback': submit_notebook_button_click
        }
      ],
      group_id);
      // Add text and set alignment for button
      $('#' + group_id).attr('style', 'float:right');
      $('#' + button_id)
        .attr('style', 'background-color:rgb(42, 115, 204); color:white; padding:4px 8px')
        .text('Submit Assignment');
    });
  };

  var add_submit_button_for_readwrite = function(has_required_metadata) {
    IPython.notebook.save_notebook().then(function() {
      var callbacks = { 'iopub' : {'output' : function(out) {
        var file_path = out.content.text;
        var is_readonly = file_path.startsWith('/home/jovyan/work/readonly');
        if (has_required_metadata && !is_readonly) {
          add_submit_button();
        }
      }}}
      var expression =
        '%%python \n' +
        'import os \n' +
        'print(os.getcwd())';
      console.log(expression);
      IPython.notebook.kernel.execute(expression, callbacks, {silent:false});
    });
  }

  var load_ipython_extension = function() {
    require(['jquery'], function ($) {
      if (!$.isEmptyObject(IPython.notebook.metadata)) {
        var coursera_metadata = IPython.notebook.metadata.coursera;
        var course_slug = coursera_metadata && coursera_metadata.course_slug;
        var launcher_item_id = coursera_metadata && coursera_metadata.launcher_item_id;
        var graded_item_id = coursera_metadata && coursera_metadata.graded_item_id;

        // Only add the submit button
        // 1. if the notebook has all of the required metadata fields, and
        // 2. if the notebook is not under readonly path
        add_submit_button_for_readwrite(course_slug && launcher_item_id && graded_item_id)
      } else {
        // Wait for notebook to load, then try again
        window.setTimeout(load_ipython_extension, 500);
      }
    });
  };

  return {
    load_ipython_extension : load_ipython_extension,
  };
});
