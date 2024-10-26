from flask import Flask, request, jsonify

class ChatServer:
    def __init__(self, model_runners):
        self.model_runners = model_runners
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            try:
                data = request.get_json()
                if 'messages' not in data or 'model' not in data:
                    return jsonify({'error': 'Invalid request format, must include messages and model fields'}), 400

                model_name = data['model']
                if model_name not in self.model_runners:
                    return jsonify({'error': f'Model {model_name} not found'}), 404

                model_runner = self.model_runners[model_name]
                response = model_runner.chat_completion(data['messages'])

                return jsonify(response), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def start_server(self, port=5042):
        self.app.run(host='0.0.0.0', port=port)
