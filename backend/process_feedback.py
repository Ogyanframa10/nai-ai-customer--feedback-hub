import boto3
import json
import os
import re
import uuid
from datetime import datetime, timezone

comprehend = boto3.client('comprehend')
table = boto3.resource('dynamodb').Table(
    os.environ.get('TABLE_NAME', 'niinai-feedback-tb')
)

EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def respond(status, body):
    # CORS is handled by API Gateway (Step 5), so no CORS headers here
    return {
        'statusCode': status,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body') or '{}')
    except ValueError:
        return respond(400, {'error': 'Invalid JSON'})

    name = str(body.get('name', '')).strip()[:100]
    email = str(body.get('email', '')).strip()[:254]
    message = str(body.get('message', '')).strip()[:1000]

    # Input validation
    if not name or not message:
        return respond(400, {'error': 'Name and message are required'})
    if not EMAIL_RE.match(email):
        return respond(400, {'error': 'Invalid email address'})

    try:
        sentiment = comprehend.detect_sentiment(Text=message, LanguageCode='en')
        entities_raw = comprehend.detect_entities(Text=message, LanguageCode='en')

        entities = [
            {'text': e['Text'], 'type': e['Type']}
            for e in entities_raw['Entities'][:10]
        ]

        label = sentiment['Sentiment']
        feedback_id = str(uuid.uuid4())

        table.put_item(Item={
            'feedback_id': feedback_id,
            'name': name,
            'email': email,
            'message': message,
            'sentiment': label,
            'sentiment_score': str(sentiment['SentimentScore'][label.title()]),
            'entities': entities,
            'submitted_at': datetime.now(timezone.utc).isoformat(),
        })

        return respond(200, {'success': True, 'id': feedback_id, 'sentiment': label})

    except Exception as e:
        print(f'Error: {e}')  # visible in CloudWatch logs
        return respond(500, {'error': 'Could not process feedback'})
