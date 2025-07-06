<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# EmoSense Backend - Copilot Instructions

This is a Python FastAPI backend project for emotion analysis. When generating code for this project, please follow these guidelines:

## Project Context
- **Framework**: FastAPI with Python 3.9+
- **Purpose**: Emotion analysis API for text, video, and audio processing
- **Architecture**: Clean architecture with separation of concerns
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis
- **Authentication**: JWT-based authentication
- **Task Processing**: Celery for async tasks

## Code Style and Standards
- Follow PEP 8 Python style guidelines
- Use type hints throughout the codebase
- Implement comprehensive error handling
- Add detailed docstrings for all functions and classes
- Use async/await for I/O operations
- Implement proper logging with structured logging

## API Design Patterns
- Use Pydantic models for request/response validation
- Implement proper HTTP status codes
- Use dependency injection for database sessions and authentication
- Follow RESTful API conventions
- Include comprehensive API documentation

## Security Best Practices
- Always validate and sanitize input data
- Implement proper authentication and authorization
- Use secure password hashing
- Validate file uploads thoroughly
- Implement rate limiting
- Never expose sensitive information in responses

## Database Patterns
- Use SQLAlchemy models with proper relationships
- Implement database migrations with Alembic
- Use connection pooling
- Implement proper transaction handling
- Add database indexes for performance

## Machine Learning Integration
- Use pre-trained models for emotion analysis
- Implement proper model loading and caching
- Handle different input formats (text, audio, video)
- Implement batch processing for efficiency
- Add model versioning support

## File Processing
- Validate file types and sizes
- Implement secure file storage
- Use temporary files for processing
- Clean up resources after processing
- Support multiple file formats

## Testing
- Write unit tests for all business logic
- Use pytest for testing framework
- Mock external dependencies
- Test API endpoints with test client
- Implement integration tests

## Performance
- Use async processing for I/O operations
- Implement caching strategies
- Use background tasks for heavy processing
- Optimize database queries
- Monitor performance metrics

## Error Handling
- Use custom exception classes
- Implement global exception handlers
- Log errors with proper context
- Return user-friendly error messages
- Handle timeouts and retries

## Documentation
- Generate OpenAPI/Swagger documentation
- Include examples in API documentation
- Document all environment variables
- Provide setup and deployment instructions
- Document API rate limits and constraints
