--
--  Standard metadata insertion
--
--
INSERT INTO
    `rocpd_metadata{{uuid}}` ("tag", "value")
VALUES
    ("schema_version", "3"),
    ("uuid", "{{uuid}}"),
    ("guid", "{{guid}}");
