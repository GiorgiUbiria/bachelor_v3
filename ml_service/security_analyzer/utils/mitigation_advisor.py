import logging
from typing import Dict, List, Optional, Any
from ..config import ATTACK_TYPES

logger = logging.getLogger(__name__)

class MitigationAdvisor:
    def __init__(self):
        self.mitigation_strategies = {
            "XSS": {
                "immediate": [
                    "Sanitize user input",
                    "Implement Content Security Policy (CSP)",
                    "Use HTML entity encoding",
                    "Validate input on server side"
                ],
                "long_term": [
                    "Implement input validation frameworks",
                    "Use parameterized queries",
                    "Regular security code reviews",
                    "Web Application Firewall (WAF) deployment"
                ],
                "severity": "high"
            },
            "SQL_INJECTION": {
                "immediate": [
                    "Use parameterized queries/prepared statements",
                    "Escape special characters",
                    "Validate input types and ranges",
                    "Apply principle of least privilege to database users"
                ],
                "long_term": [
                    "Implement ORM frameworks",
                    "Database activity monitoring",
                    "Regular SQL injection testing",
                    "Database security hardening"
                ],
                "severity": "critical"
            },
            "CSRF": {
                "immediate": [
                    "Implement CSRF tokens",
                    "Verify HTTP referer header",
                    "Use SameSite cookie attribute",
                    "Validate origin header"
                ],
                "long_term": [
                    "Implement proper session management",
                    "Use framework CSRF protection",
                    "Regular penetration testing",
                    "Security awareness training"
                ],
                "severity": "medium"
            },
            "COMMAND_INJECTION": {
                "immediate": [
                    "Avoid system calls with user input",
                    "Use safe APIs instead of shell commands",
                    "Implement input validation",
                    "Use whitelisting for allowed commands"
                ],
                "long_term": [
                    "Containerization and sandboxing",
                    "Regular security audits",
                    "Code review processes",
                    "Principle of least privilege"
                ],
                "severity": "critical"
            },
            "PATH_TRAVERSAL": {
                "immediate": [
                    "Validate file paths",
                    "Use canonical path resolution",
                    "Implement file access controls",
                    "Sanitize file names"
                ],
                "long_term": [
                    "Implement proper file system permissions",
                    "Use secure file handling libraries",
                    "Regular security assessments",
                    "Access logging and monitoring"
                ],
                "severity": "high"
            }
        }
        
    def get_mitigation_advice(self, attack_type: str, severity: str = "medium") -> Dict[str, Any]:
        """Get mitigation advice for specific attack type"""
        try:
            if attack_type not in self.mitigation_strategies:
                return self._get_generic_advice(severity)
                
            strategy = self.mitigation_strategies[attack_type]
            
            return {
                "attack_type": attack_type,
                "severity": strategy.get("severity", severity),
                "immediate_actions": strategy["immediate"],
                "long_term_strategies": strategy["long_term"],
                "priority": self._get_priority(strategy.get("severity", severity)),
                "estimated_implementation_time": self._get_implementation_time(attack_type),
                "compliance_considerations": self._get_compliance_notes(attack_type)
            }
            
        except Exception as e:
            logger.error(f"Error getting mitigation advice: {e}")
            return self._get_generic_advice(severity)
            
    def get_comprehensive_report(self, detected_threats: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive mitigation report for multiple threats"""
        try:
            if not detected_threats:
                return {"status": "no_threats", "recommendations": []}
                
            report = {
                "summary": {
                    "total_threats": len(detected_threats),
                    "critical_threats": 0,
                    "high_threats": 0,
                    "medium_threats": 0,
                    "low_threats": 0
                },
                "prioritized_actions": [],
                "long_term_strategy": [],
                "compliance_requirements": [],
                "estimated_timeline": {}
            }
            
            # Analyze each threat
            all_actions = []
            all_strategies = []
            
            for threat in detected_threats:
                attack_type = threat.get("attack_type", "UNKNOWN")
                confidence = threat.get("confidence", 0.0)
                
                advice = self.get_mitigation_advice(attack_type)
                severity = advice.get("severity", "medium")
                
                # Count threats by severity
                if severity == "critical":
                    report["summary"]["critical_threats"] += 1
                elif severity == "high":
                    report["summary"]["high_threats"] += 1
                elif severity == "medium":
                    report["summary"]["medium_threats"] += 1
                else:
                    report["summary"]["low_threats"] += 1
                
                # Collect actions
                for action in advice["immediate_actions"]:
                    if action not in all_actions:
                        all_actions.append({
                            "action": action,
                            "priority": advice["priority"],
                            "attack_types": [attack_type],
                            "confidence": confidence
                        })
                        
                for strategy in advice["long_term_strategies"]:
                    if strategy not in all_strategies:
                        all_strategies.append(strategy)
                        
            # Sort actions by priority
            priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            all_actions.sort(key=lambda x: priority_order.get(x["priority"], 1), reverse=True)
            
            report["prioritized_actions"] = all_actions[:10]  # Top 10 actions
            report["long_term_strategy"] = list(set(all_strategies))[:8]  # Top 8 strategies
            
            # Compliance considerations
            report["compliance_requirements"] = self._get_compliance_summary(detected_threats)
            
            # Timeline estimation
            report["estimated_timeline"] = self._estimate_implementation_timeline(detected_threats)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {"status": "error", "message": str(e)}
            
    def _get_generic_advice(self, severity: str) -> Dict[str, Any]:
        """Get generic security advice"""
        return {
            "attack_type": "UNKNOWN",
            "severity": severity,
            "immediate_actions": [
                "Review and validate all user inputs",
                "Implement proper authentication",
                "Enable security logging",
                "Apply security headers"
            ],
            "long_term_strategies": [
                "Regular security assessments",
                "Security awareness training",
                "Implement security frameworks",
                "Continuous monitoring"
            ],
            "priority": self._get_priority(severity),
            "estimated_implementation_time": "1-2 weeks",
            "compliance_considerations": ["General security best practices"]
        }
        
    def _get_priority(self, severity: str) -> str:
        """Convert severity to priority level"""
        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low"
        }
        return severity_map.get(severity.lower(), "medium")
        
    def _get_implementation_time(self, attack_type: str) -> str:
        """Estimate implementation time for mitigation"""
        time_estimates = {
            "XSS": "2-3 days",
            "SQL_INJECTION": "1-2 days",
            "CSRF": "1 day",
            "COMMAND_INJECTION": "3-5 days",
            "PATH_TRAVERSAL": "2-3 days"
        }
        return time_estimates.get(attack_type, "1-2 weeks")
        
    def _get_compliance_notes(self, attack_type: str) -> List[str]:
        """Get compliance considerations for attack type"""
        compliance_map = {
            "XSS": ["OWASP Top 10", "PCI DSS", "ISO 27001"],
            "SQL_INJECTION": ["OWASP Top 10", "PCI DSS", "SOX", "HIPAA"],
            "CSRF": ["OWASP Top 10", "PCI DSS"],
            "COMMAND_INJECTION": ["OWASP Top 10", "ISO 27001"],
            "PATH_TRAVERSAL": ["OWASP Top 10", "PCI DSS"]
        }
        return compliance_map.get(attack_type, ["General security standards"])
        
    def _get_compliance_summary(self, detected_threats: List[Dict]) -> List[str]:
        """Get summary of compliance requirements"""
        all_compliance = set()
        for threat in detected_threats:
            attack_type = threat.get("attack_type", "UNKNOWN")
            compliance = self._get_compliance_notes(attack_type)
            all_compliance.update(compliance)
        return list(all_compliance)
        
    def _estimate_implementation_timeline(self, detected_threats: List[Dict]) -> Dict[str, str]:
        """Estimate overall implementation timeline"""
        critical_count = sum(1 for t in detected_threats if 
                           self.mitigation_strategies.get(t.get("attack_type", ""), {}).get("severity") == "critical")
        high_count = sum(1 for t in detected_threats if 
                        self.mitigation_strategies.get(t.get("attack_type", ""), {}).get("severity") == "high")
        
        if critical_count > 0:
            immediate = "1-3 days"
            short_term = "1-2 weeks"
            long_term = "1-3 months"
        elif high_count > 0:
            immediate = "3-5 days"
            short_term = "2-4 weeks"
            long_term = "2-4 months"
        else:
            immediate = "1 week"
            short_term = "1 month"
            long_term = "3-6 months"
            
        return {
            "immediate_fixes": immediate,
            "short_term_improvements": short_term,
            "long_term_strategy": long_term
        }

# Global mitigation advisor instance
mitigation_advisor = MitigationAdvisor() 