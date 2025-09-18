# Security Architecture - Insurance Claims Fraud Detection

## Executive Summary

This document outlines the comprehensive security architecture for the insurance claims fraud detection system, ensuring protection of sensitive healthcare data (PHI/PII), compliance with regulations (HIPAA, GDPR), and defense against cyber threats while maintaining system performance and usability.

## Security Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Perimeter                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   WAF & DDoS Protection                │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │                                      │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │              API Gateway & Authentication              │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │                                      │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │           Zero Trust Network Architecture              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │   DMZ    │  │ App Tier │  │    Data Tier     │    │  │
│  │  │          │←→│          │←→│                  │    │  │
│  │  │ Public   │  │ Private  │  │ Highly Restricted│    │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Encryption & Key Management                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │      Monitoring, Logging & Incident Response          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Protection

### Data Classification

| Level | Classification | Data Types | Protection Requirements |
|-------|---------------|------------|------------------------|
| **L4** | Highly Sensitive | SSN, Medical Records, Financial Data | Full encryption, MFA, audit logging, restricted access |
| **L3** | Sensitive | PHI, PII, Provider Information | Encryption, role-based access, logging |
| **L2** | Internal | Aggregated metrics, Reports | Standard encryption, controlled access |
| **L1** | Public | Reference data, Published statistics | Basic protection, read-only |

### Encryption Strategy

#### Encryption at Rest

```yaml
encryption_at_rest:
  databases:
    method: Transparent Data Encryption (TDE)
    algorithm: AES-256-GCM
    key_rotation: 90_days

  file_storage:
    s3:
      server_side_encryption: SSE-KMS
      customer_managed_keys: true
      bucket_policies: deny_unencrypted_uploads

  backups:
    encryption: AES-256
    key_storage: separate_hsm
    key_escrow: enabled

  local_storage:
    full_disk_encryption: BitLocker/FileVault
    tpm_integration: required
```

#### Encryption in Transit

```yaml
encryption_in_transit:
  external_communications:
    protocol: TLS 1.3
    cipher_suites:
      - TLS_AES_256_GCM_SHA384
      - TLS_CHACHA20_POLY1305_SHA256
    certificate_pinning: enabled
    hsts: max_age=31536000

  internal_communications:
    service_mesh: istio
    mtls: required
    certificate_rotation: 30_days

  api_security:
    protocols: [https]
    minimum_tls: 1.2
    perfect_forward_secrecy: required
```

#### Field-Level Encryption

```python
class FieldEncryption:
    """
    Encrypt sensitive fields while preserving searchability
    """

    def __init__(self):
        self.format_preserving = FPE()
        self.deterministic = DeterministicEncryption()
        self.randomized = RandomizedEncryption()

    def encrypt_field(self, field_name, value, searchable=False):
        if field_name == 'ssn':
            # Format-preserving encryption for SSN
            return self.format_preserving.encrypt(value, format='###-##-####')

        elif field_name in ['patient_name', 'dob']:
            # Deterministic for searchability
            if searchable:
                return self.deterministic.encrypt(value)
            else:
                return self.randomized.encrypt(value)

        elif field_name == 'medical_notes':
            # Always randomized for free text
            return self.randomized.encrypt(value)

        return value
```

## Identity & Access Management

### Authentication Architecture

```yaml
authentication:
  methods:
    primary:
      type: oauth2_with_oidc
      provider: [okta, azure_ad]
      token_lifetime: 1_hour
      refresh_token: 8_hours

    mfa:
      required_for: [admin, phi_access, financial_data]
      methods:
        - authenticator_app
        - hardware_token
        - biometric
      bypass: break_glass_procedure

  service_accounts:
    authentication: client_certificates
    rotation: 30_days
    audit: all_usage

  api_keys:
    rotation: 90_days
    encryption: vault_stored
    rate_limiting: per_key_limits
```

### Authorization Model

```python
class RBACWithABAC:
    """
    Role-Based Access Control with Attribute-Based policies
    """

    def __init__(self):
        self.roles = {
            'fraud_investigator': {
                'permissions': ['read_claims', 'read_alerts', 'update_investigation'],
                'data_access': 'masked_phi',
                'time_restriction': 'business_hours'
            },
            'data_scientist': {
                'permissions': ['read_anonymized_data', 'run_models'],
                'data_access': 'de_identified',
                'environment': 'development'
            },
            'system_admin': {
                'permissions': ['full_system_access'],
                'data_access': 'audit_only',
                'requires_approval': True
            }
        }

        self.attributes = {
            'location': ['us', 'eu'],
            'department': ['fraud', 'analytics', 'it'],
            'clearance_level': [1, 2, 3, 4],
            'training_completed': ['hipaa', 'gdpr', 'security']
        }

    def check_access(self, user, resource, action, context):
        # Check role-based permissions
        role_allowed = self.check_role_permission(user.role, action)

        # Check attribute-based policies
        attr_allowed = self.check_attributes(user.attributes, resource, context)

        # Check dynamic policies
        dynamic_allowed = self.check_dynamic_policies(user, resource, context)

        return role_allowed and attr_allowed and dynamic_allowed
```

### Privileged Access Management

```yaml
pam_configuration:
  privileged_accounts:
    identification:
      - admin_accounts
      - service_accounts
      - database_accounts
      - cloud_root_accounts

  controls:
    just_in_time_access:
      max_duration: 4_hours
      approval_required: true
      automatic_revocation: true

    session_management:
      recording: all_sessions
      real_time_monitoring: true
      command_filtering: enabled

    password_vault:
      solution: hashicorp_vault
      rotation: 30_days
      checkout_process: required
      break_glass: documented

  monitoring:
    alerts:
      - unusual_access_patterns
      - after_hours_access
      - failed_authentication
      - privilege_escalation
```

## Network Security

### Zero Trust Architecture

```yaml
zero_trust:
  principles:
    - never_trust_always_verify
    - least_privilege_access
    - assume_breach
    - verify_explicitly

  implementation:
    micro_segmentation:
      technology: software_defined_perimeter
      segments:
        - dmz
        - application_tier
        - data_tier
        - management_tier

    identity_verification:
      device_trust: required
      user_verification: continuous
      application_verification: certificate_based

    traffic_inspection:
      east_west: all_traffic
      north_south: all_traffic
      encrypted_traffic: ssl_inspection
```

### Network Segmentation

```
┌─────────────────────────────────────────────────────┐
│                  Internet                           │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   WAF/CDN         │
         │  (CloudFlare)     │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Load Balancer    │
         │   (Public)        │
         └─────────┬─────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│               DMZ (10.1.0.0/24)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  Web       │  │   API      │  │   Proxy    │   │
│  │  Servers   │  │  Gateway   │  │   Server   │   │
│  └────────────┘  └────────────┘  └────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │ Firewall
┌──────────────────▼──────────────────────────────────┐
│         Application Tier (10.2.0.0/24)              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │   App      │  │   ML       │  │   Cache    │   │
│  │  Servers   │  │  Services  │  │   Layer    │   │
│  └────────────┘  └────────────┘  └────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │ Firewall
┌──────────────────▼──────────────────────────────────┐
│            Data Tier (10.3.0.0/24)                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  Database  │  │   Data     │  │   Backup   │   │
│  │  Cluster   │  │   Lake     │  │   Storage  │   │
│  └────────────┘  └────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────┘
```

### Firewall Rules

```yaml
firewall_rules:
  dmz_ingress:
    - source: 0.0.0.0/0
      destination: dmz
      ports: [443]
      protocol: tcp
      action: allow

  dmz_to_app:
    - source: dmz
      destination: app_tier
      ports: [8080, 8443]
      protocol: tcp
      action: allow

  app_to_data:
    - source: app_tier
      destination: data_tier
      ports: [5432, 6379, 9200]
      protocol: tcp
      action: allow

  default:
    action: deny
    logging: true
```

## Application Security

### Secure Development Lifecycle

```yaml
sdlc_security:
  design_phase:
    - threat_modeling
    - security_requirements
    - architecture_review

  development_phase:
    - secure_coding_training
    - peer_code_review
    - static_analysis

  testing_phase:
    - sast_scanning
    - dast_scanning
    - dependency_scanning
    - penetration_testing

  deployment_phase:
    - security_gates
    - configuration_review
    - runtime_protection

  maintenance_phase:
    - patch_management
    - vulnerability_monitoring
    - incident_response
```

### Input Validation

```python
class InputValidator:
    """
    Comprehensive input validation for all user inputs
    """

    def __init__(self):
        self.validators = {
            'claim_id': self.validate_claim_id,
            'npi': self.validate_npi,
            'ssn': self.validate_ssn,
            'amount': self.validate_amount,
            'date': self.validate_date,
            'diagnosis_code': self.validate_diagnosis_code
        }

    def validate_claim_id(self, value):
        pattern = r'^CLM-\d{4}-\d{6}$'
        if not re.match(pattern, value):
            raise ValidationError('Invalid claim ID format')

        # Check for SQL injection
        if self.contains_sql_injection(value):
            raise SecurityError('Potential SQL injection detected')

        return value

    def validate_amount(self, value):
        try:
            amount = Decimal(str(value))
            if amount < 0 or amount > 1000000:
                raise ValidationError('Amount out of valid range')
            return amount
        except:
            raise ValidationError('Invalid amount format')

    def contains_sql_injection(self, value):
        sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE)\b)',
            r'(--|#|\/\*|\*\/)',
            r'(\bOR\b.*=.*)',
            r'(\bAND\b.*=.*)'
        ]

        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
```

### API Security

```yaml
api_security:
  authentication:
    type: oauth2
    token_validation: jwt_with_signature
    token_storage: secure_cookie

  authorization:
    type: scope_based
    policy_engine: opa
    fine_grained: true

  rate_limiting:
    global: 10000_per_minute
    per_user: 100_per_minute
    per_ip: 1000_per_minute
    ddos_protection: cloudflare

  input_validation:
    schema_validation: openapi_3.0
    content_type_validation: strict
    size_limits:
      request_body: 10_mb
      file_upload: 100_mb

  security_headers:
    x_frame_options: DENY
    x_content_type_options: nosniff
    x_xss_protection: 1; mode=block
    content_security_policy: strict
    strict_transport_security: max-age=31536000

  cors:
    allowed_origins: [https://app.example.com]
    allowed_methods: [GET, POST, PUT, DELETE]
    max_age: 3600
```

## Compliance & Privacy

### HIPAA Compliance

```yaml
hipaa_compliance:
  administrative_safeguards:
    - security_officer_designation
    - workforce_training
    - access_management
    - incident_response_plan
    - business_associate_agreements

  physical_safeguards:
    - facility_access_controls
    - workstation_security
    - device_and_media_controls

  technical_safeguards:
    - access_controls:
        unique_user_identification: true
        automatic_logoff: 15_minutes
        encryption_decryption: required

    - audit_controls:
        log_all_phi_access: true
        log_retention: 7_years
        log_integrity: tamper_proof

    - integrity_controls:
        data_validation: true
        error_correction: true

    - transmission_security:
        encryption: required
        integrity_checks: required
```

### GDPR Compliance

```yaml
gdpr_compliance:
  lawful_basis:
    - legitimate_interest
    - consent_management
    - contract_fulfillment

  data_subject_rights:
    right_to_access:
      response_time: 30_days
      format: machine_readable

    right_to_erasure:
      implementation: automated
      exceptions: legal_obligations

    right_to_portability:
      format: json
      transfer_mechanism: secure_api

    right_to_rectification:
      user_interface: self_service
      audit_trail: required

  privacy_by_design:
    - data_minimization
    - purpose_limitation
    - default_privacy_settings
    - end_to_end_security

  breach_notification:
    supervisory_authority: 72_hours
    data_subjects: without_undue_delay
```

### Data Anonymization

```python
class DataAnonymizer:
    """
    Anonymize sensitive data for analytics
    """

    def __init__(self):
        self.k_anonymity = KAnonymityEngine(k=5)
        self.differential_privacy = DifferentialPrivacy(epsilon=1.0)

    def anonymize_dataset(self, data):
        # Remove direct identifiers
        data = self.remove_identifiers(data)

        # Apply k-anonymity
        data = self.k_anonymity.anonymize(
            data,
            quasi_identifiers=['age', 'zip_code', 'gender'],
            sensitive_attributes=['diagnosis', 'treatment']
        )

        # Add noise for differential privacy
        data = self.differential_privacy.add_noise(data)

        # Generate synthetic records
        data = self.add_synthetic_records(data, ratio=0.1)

        return data

    def remove_identifiers(self, data):
        identifiers = ['ssn', 'name', 'address', 'phone', 'email']
        return data.drop(columns=identifiers, errors='ignore')
```

## Security Monitoring

### SIEM Integration

```yaml
siem_configuration:
  platform: splunk
  data_sources:
    - application_logs
    - infrastructure_logs
    - security_events
    - database_audit_logs
    - api_access_logs
    - network_flow_logs

  correlation_rules:
    - multiple_failed_logins
    - unusual_data_access_patterns
    - privilege_escalation_attempts
    - data_exfiltration_indicators
    - anomalous_api_usage

  alerting:
    critical:
      - data_breach_detected
      - ransomware_indicators
      - admin_account_compromise

    high:
      - suspicious_user_behavior
      - unauthorized_access_attempt
      - configuration_changes

    medium:
      - policy_violations
      - compliance_deviations
```

### Threat Detection

```python
class ThreatDetectionEngine:
    """
    Real-time threat detection using ML
    """

    def __init__(self):
        self.anomaly_detector = IsolationForest()
        self.pattern_matcher = PatternMatcher()
        self.behavior_analyzer = UserBehaviorAnalytics()

    def detect_threats(self, event_stream):
        threats = []

        for event in event_stream:
            # Check for known attack patterns
            if self.pattern_matcher.match(event):
                threats.append({
                    'type': 'known_pattern',
                    'severity': 'high',
                    'event': event
                })

            # Detect anomalies
            if self.anomaly_detector.is_anomaly(event):
                threats.append({
                    'type': 'anomaly',
                    'severity': 'medium',
                    'event': event
                })

            # Analyze user behavior
            risk_score = self.behavior_analyzer.calculate_risk(event)
            if risk_score > 0.8:
                threats.append({
                    'type': 'behavioral',
                    'severity': 'high',
                    'event': event,
                    'risk_score': risk_score
                })

        return threats
```

## Incident Response

### Incident Response Plan

```yaml
incident_response:
  phases:
    preparation:
      - incident_response_team
      - communication_plan
      - tool_deployment
      - training_exercises

    detection_analysis:
      - alert_triage
      - incident_classification
      - impact_assessment
      - evidence_collection

    containment:
      short_term:
        - isolate_affected_systems
        - block_malicious_traffic
        - disable_compromised_accounts

      long_term:
        - patch_vulnerabilities
        - improve_security_controls
        - update_detection_rules

    eradication:
      - remove_malware
      - close_vulnerabilities
      - reset_credentials

    recovery:
      - restore_systems
      - monitor_for_reinfection
      - validate_functionality

    lessons_learned:
      - incident_review
      - update_procedures
      - improve_controls

  severity_levels:
    critical:
      response_time: 15_minutes
      escalation: ciso_ceo

    high:
      response_time: 1_hour
      escalation: security_manager

    medium:
      response_time: 4_hours
      escalation: team_lead

    low:
      response_time: 24_hours
      escalation: analyst
```

### Forensics Capabilities

```yaml
forensics:
  evidence_collection:
    - memory_dumps
    - disk_images
    - network_captures
    - log_files
    - configuration_files

  analysis_tools:
    - volatility
    - wireshark
    - elk_stack
    - autopsy

  chain_of_custody:
    documentation: required
    hash_verification: sha256
    secure_storage: encrypted_vault
```

## Vulnerability Management

### Vulnerability Scanning

```yaml
vulnerability_scanning:
  infrastructure:
    frequency: weekly
    tools: [nessus, qualys]
    scope: all_systems

  applications:
    sast:
      frequency: every_commit
      tools: [sonarqube, checkmarx]

    dast:
      frequency: daily
      tools: [owasp_zap, burp_suite]

    dependency_scanning:
      frequency: every_build
      tools: [snyk, dependabot]

  remediation:
    critical: 24_hours
    high: 7_days
    medium: 30_days
    low: 90_days
```

## Security Testing

### Penetration Testing

```yaml
penetration_testing:
  frequency:
    external: quarterly
    internal: bi_annually
    red_team: annually

  scope:
    - network_infrastructure
    - web_applications
    - api_endpoints
    - mobile_applications
    - social_engineering

  methodology:
    - owasp_testing_guide
    - ptes_standard
    - nist_framework

  reporting:
    - executive_summary
    - technical_details
    - remediation_roadmap
    - retest_validation
```

## Business Continuity

### Disaster Recovery

```yaml
disaster_recovery:
  rto: 4_hours
  rpo: 1_hour

  backup_strategy:
    frequency:
      full: weekly
      incremental: daily
      transaction_log: 15_minutes

    storage:
      onsite: immediate_recovery
      offsite: geographic_redundancy
      cloud: long_term_retention

    testing:
      restore_test: monthly
      full_dr_drill: quarterly

  failover:
    automatic: true
    monitoring: 24x7
    rollback_capability: true
```

## Security Metrics

```yaml
security_metrics:
  kpis:
    - mean_time_to_detect: target: <1_hour
    - mean_time_to_respond: target: <4_hours
    - vulnerability_remediation_rate: target: 95%
    - security_training_completion: target: 100%
    - patch_compliance_rate: target: 98%

  reporting:
    frequency: monthly
    audience: [ciso, board, stakeholders]
    dashboard: real_time
```

## Conclusion

This security architecture provides comprehensive protection for the insurance claims fraud detection system, ensuring data confidentiality, integrity, and availability while maintaining regulatory compliance and defending against evolving cyber threats.