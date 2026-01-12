from pydantic import BaseModel, Field

class CleanedProduct(BaseModel):
    """Schema for cleaned product data from the VLM."""
    clean_title: str = Field(..., description="Cleaned product title")
    style: str = Field(..., description="Furniture style")
    material: str = Field(..., description="Primary material")
    color: str = Field(..., description="Primary color")
    object_type: str = Field(..., description="Type of furniture")