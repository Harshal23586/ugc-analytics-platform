"""
API Endpoints for UGC/AICTE Institutional Analytics Platform
Provides RESTful API for system integration and data access
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uvicorn
import json
import sqlite3
import pandas as pd
from datetime import datetime
import os
from typing import List

# Import existing analyzer
from app_V3_Very_Good import InstitutionalAIAnalyzer

app = FastAPI(
    title="UGC/AICTE Institutional Analytics API",
    description="API for accessing institutional performance data and AI analytics",
    version="2.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
API_KEYS = {
    "ugc_admin": "ugc_api_key_2024_secure",
    "aictel_team": "aictel_api_2024_secure",
    "hackathon_2024": "smart_india_hackathon_key",
    "institution_api": "institution_access_2024"
}

# Initialize analyzer
analyzer = None

class StartupEvent:
    async def __init__(self):
        global analyzer
        analyzer = InstitutionalAIAnalyzer()

app.add_event_handler("startup", StartupEvent().__init__)

# Pydantic Models
class InstitutionBase(BaseModel):
    institution_id: str
    institution_name: str
    institution_type: str
    state: str
    year: int

class InstitutionPerformance(InstitutionBase):
    performance_score: float
    naac_grade: Optional[str] = None
    nirf_ranking: Optional[int] = None
    placement_rate: float
    risk_level: str
    approval_recommendation: str

class PerformanceMetrics(BaseModel):
    academic_excellence: Dict
    research_innovation: Dict
    infrastructure_facilities: Dict
    governance_administration: Dict
    student_development: Dict
    social_impact: Dict

class DocumentRequirement(BaseModel):
    approval_type: str
    mandatory_documents: List[str]
    supporting_documents: List[str]

class AIAnalysisRequest(BaseModel):
    institution_id: str
    documents: List[str]  # Base64 encoded or URLs
    analysis_type: str = "comprehensive"

class AIAnalysisResponse(BaseModel):
    institution_id: str
    analysis_type: str
    performance_score: float
    risk_assessment: Dict
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence_score: float

# Authentication middleware
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in API_KEYS.values():
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return token

# Health check endpoint
@app.get("/")
async def root():
    return {
        "service": "UGC/AICTE Institutional Analytics API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/docs - API Documentation",
            "/institutions - Get all institutions",
            "/performance/{institution_id} - Get performance data",
            "/analysis - Run AI analysis",
            "/metrics - Get performance metrics configuration",
            "/documents - Get document requirements"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Institution Data Endpoints
@app.get("/institutions", response_model=List[InstitutionBase])
async def get_institutions(
    year: Optional[int] = Query(None, description="Filter by year"),
    institution_type: Optional[str] = Query(None, description="Filter by type"),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(100, description="Limit results"),
    api_key: str = Depends(verify_api_key)
):
    """Get list of institutions with basic information"""
    try:
        df = analyzer.historical_data
        
        if year:
            df = df[df['year'] == year]
        if institution_type:
            df = df[df['institution_type'] == institution_type]
        if state:
            df = df[df['state'] == state]
        
        df = df.head(limit)
        
        institutions = []
        for _, row in df.iterrows():
            institutions.append({
                "institution_id": row['institution_id'],
                "institution_name": row['institution_name'],
                "institution_type": row['institution_type'],
                "state": row['state'],
                "year": row['year']
            })
        
        return institutions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/institutions/{institution_id}", response_model=List[InstitutionPerformance])
async def get_institution_performance(
    institution_id: str,
    start_year: Optional[int] = Query(None),
    end_year: Optional[int] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """Get performance data for a specific institution"""
    try:
        df = analyzer.historical_data
        df = df[df['institution_id'] == institution_id]
        
        if start_year:
            df = df[df['year'] >= start_year]
        if end_year:
            df = df[df['year'] <= end_year]
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Institution not found")
        
        performances = []
        for _, row in df.iterrows():
            performances.append({
                "institution_id": row['institution_id'],
                "institution_name": row['institution_name'],
                "institution_type": row['institution_type'],
                "state": row['state'],
                "year": row['year'],
                "performance_score": float(row['performance_score']),
                "naac_grade": row.get('naac_grade'),
                "nirf_ranking": row.get('nirf_ranking'),
                "placement_rate": float(row.get('placement_rate', 0)),
                "risk_level": row.get('risk_level', 'Unknown'),
                "approval_recommendation": row.get('approval_recommendation', 'Pending')
            })
        
        return performances
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/institutions/{institution_id}/current", response_model=InstitutionPerformance)
async def get_current_performance(institution_id: str, api_key: str = Depends(verify_api_key)):
    """Get current year performance for an institution"""
    try:
        current_year = analyzer.historical_data['year'].max()
        df = analyzer.historical_data[
            (analyzer.historical_data['institution_id'] == institution_id) & 
            (analyzer.historical_data['year'] == current_year)
        ]
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Institution not found for current year")
        
        row = df.iloc[0]
        
        return {
            "institution_id": row['institution_id'],
            "institution_name": row['institution_name'],
            "institution_type": row['institution_type'],
            "state": row['state'],
            "year": row['year'],
            "performance_score": float(row['performance_score']),
            "naac_grade": row.get('naac_grade'),
            "nirf_ranking": row.get('nirf_ranking'),
            "placement_rate": float(row.get('placement_rate', 0)),
            "risk_level": row.get('risk_level', 'Unknown'),
            "approval_recommendation": row.get('approval_recommendation', 'Pending')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Performance Analytics Endpoints
@app.get("/performance/ranking")
async def get_performance_ranking(
    year: Optional[int] = Query(None),
    institution_type: Optional[str] = Query(None),
    limit: int = Query(20),
    api_key: str = Depends(verify_api_key)
):
    """Get performance ranking of institutions"""
    try:
        df = analyzer.historical_data
        
        if year:
            df = df[df['year'] == year]
        else:
            # Default to current year
            current_year = df['year'].max()
            df = df[df['year'] == current_year]
        
        if institution_type:
            df = df[df['institution_type'] == institution_type]
        
        # Sort by performance score
        df = df.sort_values('performance_score', ascending=False).head(limit)
        
        ranking = []
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            ranking.append({
                "rank": rank,
                "institution_id": row['institution_id'],
                "institution_name": row['institution_name'],
                "performance_score": float(row['performance_score']),
                "naac_grade": row.get('naac_grade'),
                "placement_rate": float(row.get('placement_rate', 0)),
                "risk_level": row.get('risk_level', 'Unknown')
            })
        
        return {
            "year": df['year'].iloc[0] if not df.empty else year,
            "total_institutions": len(df),
            "ranking": ranking
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/comparison")
async def compare_institutions(
    institution_ids: List[str] = Query(..., description="List of institution IDs to compare"),
    year: int = Query(None, description="Year for comparison"),
    api_key: str = Depends(verify_api_key)
):
    """Compare multiple institutions"""
    try:
        df = analyzer.historical_data
        if year:
            df = df[df['year'] == year]
        else:
            current_year = df['year'].max()
            df = df[df['year'] == current_year]
        
        comparison_data = []
        
        for inst_id in institution_ids:
            inst_data = df[df['institution_id'] == inst_id]
            if not inst_data.empty:
                row = inst_data.iloc[0]
                comparison_data.append({
                    "institution_id": inst_id,
                    "institution_name": row['institution_name'],
                    "performance_score": float(row['performance_score']),
                    "naac_grade": row.get('naac_grade'),
                    "nirf_ranking": row.get('nirf_ranking'),
                    "placement_rate": float(row.get('placement_rate', 0)),
                    "student_faculty_ratio": float(row.get('student_faculty_ratio', 0)),
                    "research_publications": int(row.get('research_publications', 0)),
                    "risk_level": row.get('risk_level', 'Unknown'),
                    "approval_recommendation": row.get('approval_recommendation', 'Pending')
                })
        
        if not comparison_data:
            raise HTTPException(status_code=404, detail="No institutions found")
        
        return {
            "comparison_year": df['year'].iloc[0] if not df.empty else year,
            "institutions_compared": len(comparison_data),
            "data": comparison_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Analysis Endpoints
@app.post("/analysis/run", response_model=AIAnalysisResponse)
async def run_ai_analysis(request: AIAnalysisRequest, api_key: str = Depends(verify_api_key)):
    """Run AI analysis on institutional data"""
    try:
        # Get institution data
        df = analyzer.historical_data
        current_year = df['year'].max()
        inst_data = df[
            (df['institution_id'] == request.institution_id) & 
            (df['year'] == current_year)
        ]
        
        if inst_data.empty:
            raise HTTPException(status_code=404, detail="Institution not found")
        
        row = inst_data.iloc[0]
        
        # Generate AI insights (simplified version)
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze performance
        if row['performance_score'] >= 8.0:
            strengths.append("High performing institution")
        elif row['performance_score'] < 6.0:
            weaknesses.append("Below average performance")
            recommendations.append("Implement performance improvement plan")
        
        # Analyze specific metrics
        if row.get('placement_rate', 0) > 80:
            strengths.append("Excellent placement record")
        elif row.get('placement_rate', 0) < 60:
            weaknesses.append("Low placement rate")
            recommendations.append("Strengthen industry partnerships")
        
        if row.get('research_publications', 0) > 50:
            strengths.append("Strong research output")
        elif row.get('research_publications', 0) < 10:
            weaknesses.append("Limited research activity")
            recommendations.append("Enhance research culture and funding")
        
        # Risk assessment
        risk_score = 10 - row['performance_score']
        if risk_score < 3:
            risk_level = "Low"
        elif risk_score < 6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return AIAnalysisResponse(
            institution_id=request.institution_id,
            analysis_type=request.analysis_type,
            performance_score=float(row['performance_score']),
            risk_assessment={
                "score": risk_score,
                "level": risk_level,
                "factors": weaknesses if weaknesses else ["No significant risk factors"]
            },
            strengths=strengths if strengths else ["No specific strengths identified"],
            weaknesses=weaknesses if weaknesses else ["No major weaknesses identified"],
            recommendations=recommendations if recommendations else ["Continue current performance"],
            confidence_score=0.85
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Endpoints
@app.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics(api_key: str = Depends(verify_api_key)):
    """Get performance metrics configuration"""
    return analyzer.performance_metrics

@app.get("/documents", response_model=List[DocumentRequirement])
async def get_document_requirements(
    approval_type: Optional[str] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """Get document requirements for approval processes"""
    requirements = []
    
    for req_type, req_data in analyzer.document_requirements.items():
        if approval_type and req_type != approval_type:
            continue
            
        requirements.append({
            "approval_type": req_type,
            "mandatory_documents": req_data.get('mandatory', []),
            "supporting_documents": req_data.get('supporting', [])
        })
    
    return requirements

# Data Export Endpoints
@app.get("/export/institutions")
async def export_institutions_data(
    format: str = Query("json", regex="^(json|csv)$"),
    year: Optional[int] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """Export institutions data"""
    try:
        df = analyzer.historical_data
        
        if year:
            df = df[df['year'] == year]
        
        if format == "csv":
            csv_data = df.to_csv(index=False)
            return {
                "format": "csv",
                "data": csv_data,
                "filename": f"institutions_export_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        else:
            return {
                "format": "json",
                "data": df.to_dict(orient='records'),
                "total_records": len(df)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Statistics Endpoints
@app.get("/statistics/summary")
async def get_system_statistics(api_key: str = Depends(verify_api_key)):
    """Get system statistics summary"""
    try:
        df = analyzer.historical_data
        current_year = df['year'].max()
        current_data = df[df['year'] == current_year]
        
        return {
            "total_institutions": df['institution_id'].nunique(),
            "total_records": len(df),
            "years_covered": f"{df['year'].min()} - {df['year'].max()}",
            "current_year": current_year,
            "current_year_stats": {
                "institutions": len(current_data),
                "avg_performance": float(current_data['performance_score'].mean()),
                "avg_placement": float(current_data['placement_rate'].mean()),
                "high_performers": len(current_data[current_data['performance_score'] >= 8.0]),
                "low_performers": len(current_data[current_data['performance_score'] < 6.0])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch Operations
@app.post("/batch/analysis")
async def batch_analysis(
    institution_ids: List[str],
    analysis_type: str = "performance",
    api_key: str = Depends(verify_api_key)
):
    """Run analysis on multiple institutions"""
    try:
        results = []
        df = analyzer.historical_data
        current_year = df['year'].max()
        
        for inst_id in institution_ids:
            inst_data = df[
                (df['institution_id'] == inst_id) & 
                (df['year'] == current_year)
            ]
            
            if not inst_data.empty:
                row = inst_data.iloc[0]
                results.append({
                    "institution_id": inst_id,
                    "institution_name": row['institution_name'],
                    "performance_score": float(row['performance_score']),
                    "risk_level": row.get('risk_level', 'Unknown'),
                    "approval_status": row.get('approval_recommendation', 'Pending')
                })
        
        return {
            "total_processed": len(results),
            "failed": len(institution_ids) - len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
