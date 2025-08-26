from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import Optional
import logging
from dotenv import load_dotenv
from services.chatbot_service import ChatbotService

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Expert Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot service
chatbot_service = ChatbotService()

class ChatRequest(BaseModel):
    message: str
    expert: str

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages with expert filtering.
    """
    try:
        logger.info(f"Processing chat request for expert: {request.expert}")
        
        response = await chatbot_service.get_response(
            query=request.message,
            expert=request.expert
        )
        
        return ChatResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return ChatResponse(
            response="I'm sorry, I encountered an error while processing your question.",
            success=False,
            error=str(e)
        )

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    expert: str = Form(...)
):
    """
    Upload PDF endpoint that processes and stores documents in expert's knowledge base.
    """
    try:
        logger.info(f"Starting PDF upload process - File: {file.filename}, Expert: {expert}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type attempted: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        logger.info(f"File validation passed for: {file.filename}")
        
        # Create temporary file
        logger.info("Creating temporary file for processing...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            logger.info("Reading file content...")
            content = await file.read()
            logger.info(f"File content read successfully - Size: {len(content)} bytes")
            
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Temporary file created at: {temp_file_path}")
        
        try:
            # Process the PDF
            logger.info("Starting PDF processing with chatbot service...")
            success = await chatbot_service.upload_pdf(
                file_path=temp_file_path,
                pdf_name=file.filename,
                expert=expert
            )
            
            if success:
                logger.info(f"PDF processing completed successfully for {file.filename}")
                return {
                    "message": f"PDF '{file.filename}' uploaded successfully for expert '{expert}'",
                    "success": True
                }
            else:
                logger.error(f"PDF processing failed for {file.filename}")
                raise HTTPException(status_code=500, detail="Failed to process PDF")
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Cleaning up temporary file: {temp_file_path}")
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up successfully")
            else:
                logger.warning(f"Temporary file not found for cleanup: {temp_file_path}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    try:
        health_status = await chatbot_service.health_check()
        return {
            "status": "healthy" if health_status["overall_health"] else "unhealthy",
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/experts")
async def get_experts():
    """
    Get list of available experts.
    """
    experts = [
        "Dr. Sarah Johnson - Crop Disease Specialist",
        "Prof. Mike Chen - Soil Science Expert", 
        "Dr. Emily Rodriguez - Sustainable Farming Advisor",
        "Dr. James Wilson - Livestock Nutrition Specialist",
        "Dr. Lisa Thompson - Irrigation Management Expert"
    ]
    
    return {"experts": experts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)