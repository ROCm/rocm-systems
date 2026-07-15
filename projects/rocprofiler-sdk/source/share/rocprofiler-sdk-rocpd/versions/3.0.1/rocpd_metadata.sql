--
--  Standard metadata insertion
--
--
INSERT INTO
    `rocpd_metadata{{uuid}}` ("tag", "value")
VALUES
    ("schema_version", "{{schema_version}}"), -- full version string for the current version
    ("schema_version_major", "{{schema_version_major}}"), -- major ID for the current version
    ("schema_version_minor", "{{schema_version_minor}}"), -- minor ID for the current version
    ("schema_version_patch", "{{schema_version_patch}}"), -- patch ID for the current version
    ("schema_creation_time", CURRENT_TIMESTAMP), -- time of schema creation
    ("uuid", "{{uuid}}"),
    ("guid", "{{guid}}");
