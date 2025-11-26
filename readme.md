# Housing Support Assessment Service

A citizen-facing self-serve tool for housing vulnerability assessment, designed to help identify individuals who may need priority housing support.

## Overview

This Streamlit application provides a professional, accessible interface for citizens to complete a housing needs assessment. The system uses a heuristic model to assess risk across four key vulnerability domains:

1. **Financial Vulnerability** - Low income, unemployment, debt
2. **Care Experience** - Children in care, care leavers
3. **Institutional Discharge** - Hospital, prison, mental health facilities, armed forces
4. **Health Vulnerabilities** - Physical health, mental health, substance use, mobility needs

## Features

- **User-friendly design** - Clean, accessible interface suitable for all users
- **Progressive disclosure** - Questions adapt based on previous answers
- **Risk scoring** - Automated assessment across multiple vulnerability categories
- **Personalised responses** - LLM-style contextual feedback based on circumstances
- **Service recommendations** - Automatic matching to relevant support services
- **Mobile responsive** - Works on all device sizes

## Installation

```bash
# Clone or download the files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run housing_support_app.py
```

## Running the Application

```bash
streamlit run housing_support_app.py
```

The app will open in your browser at `http://localhost:8501`

## Assessment Categories

### Financial Circumstances
- Employment status
- Income level
- Debt situation
- Benefits status

### Care Experience
- Care history (child/young person)
- Care leaver status
- Personal Adviser support
- Children in care

### Institutional Discharge
- Hospital discharge
- Mental health facility
- Prison/young offenders
- Rehabilitation
- Armed forces

### Health & Wellbeing
- Physical health conditions
- Mental health
- Substance use support
- Mobility/accessibility needs

## Risk Assessment Model

The system calculates risk scores based on weighted responses:

| Risk Level | Score Range | Response Time |
|------------|-------------|---------------|
| Low | 0-7 | Standard processing |
| Medium | 8-13 | 3 working days |
| High | 14-19 | 24 hours |
| Critical | 20+ | Same day |

## Customisation

### Adding Questions

Edit the `QUESTIONS` list in `housing_support_app.py`:

```python
{
    "id": "unique_id",
    "section": "section_name",
    "type": "radio|text|textarea",
    "question": "Question text",
    "help": "Help text",
    "options": ["Option 1", "Option 2"],  # for radio type
    "risk_weights": {
        "Option 1": 0,
        "Option 2": 3
    }
}
```

### Conditional Questions

Add conditional logic to show questions based on previous answers:

```python
"conditional": {
    "question_id": "previous_question_id",
    "show_if_not": ["Answer to hide for"]
}
```

## Integration Points

This front-end is designed to integrate with:

1. **Backend Agent** - Send assessment data to an AI agent for deeper analysis
2. **Case Management System** - Create cases based on risk level
3. **Service Directory** - Dynamic service recommendations based on location
4. **Notification System** - Alerts for high-priority cases

## Future Enhancements

- [ ] Deep dive into Domestic Violence pathways
- [ ] Integration with local authority systems
- [ ] Multi-language support
- [ ] Accessibility audit (WCAG 2.1 AA)
- [ ] Real-time chat support integration

## License

For internal use only.

## Support

For technical issues, contact the development team.