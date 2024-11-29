import logging
import pathlib
from flask import render_template, request, redirect, send_from_directory, url_for
from flask_talisman import Talisman
import os
from helpers import MyAPIError

from helpers import MyAPIError
from app_base import app, bp
import app_oauth
import app_api

thisdir = pathlib.Path(__file__).parent.absolute()

if os.getenv('FLASK_ENV') == 'production':
    Talisman(app, content_security_policy=None)

class AppError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message

# error page
@bp.errorhandler(404)
def page_not_found(e):
    logging.warning('Page not found: %s', request.path)
    return render_template('error.html', error_code=404, error_message='Page not found'), 404

# error pages for ApiError, AppError, and Exception
@bp.errorhandler(MyAPIError)
def handle_api_error(e: MyAPIError):
    logging.exception(e)
    return render_template('error.html', error_code=e.status_code, error_message=e.message), e.status_code

@bp.errorhandler(AppError)
def handle_app_error(e: AppError):
    logging.exception(e)
    return render_template('error.html', error_code=e.status_code, error_message=e.message), e.status_code

@bp.errorhandler(Exception)
def handle_exception(e: Exception):
    logging.exception(e)
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500

# favicon route - in static/img/favicon.ico
@bp.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(bp.root_path, 'static/img'), 'favicon.ico')

@bp.route('/')
def index():
    return redirect(url_for('kubishi.builder'))

@bp.route('/builder')
def builder():
    return render_template('builder.html')

@bp.route('/translator')
def translator():
    return render_template('translator.html')

app.register_blueprint(bp)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5000)